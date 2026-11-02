#!/usr/bin/env python3
"""
센서 데이터 수집기

  MQTT 구독 → config.yaml 규칙으로 파싱 → SQLite 저장(중복 제거)
           → (선택) PostgreSQL 전송 / (선택) InfluxDB 전송

실행 방법
  python3 collector.py                      # 상시 수집
  python3 collector.py --replay sample.txt  # mosquitto_sub -v 로 저장한 파일을 재생 (테스트용)
  python3 collector.py --reprocess          # 저장된 원본을 현재 규칙으로 다시 처리 (규칙을 바꾼 뒤)

설계 메모
  - 시각은 두 가지를 모두 저장합니다.
      ts          : 메시지 안의 시각(규칙에서 고른 것). 분석·커버율 기준
      received_at : 수집기가 받은 시각. 지연·몰려 도착 여부 확인용
  - 원본 메시지를 함께 저장하므로, 규칙을 고친 뒤 --reprocess 로 과거 데이터도 다시 정리할 수 있습니다.
  - PostgreSQL·InfluxDB 로는 '전송 대기열' 방식으로 보냅니다. DB가 끊겨 있어도 로컬에 쌓아 두었다가
    연결되면 밀린 것부터 순서대로 보냅니다. 두 DB 의 전송 여부는 따로 기록합니다(pg_sent / sent).
  - PostgreSQL 에는 (space, point, device, metric, ts) 가 같은 행이 두 번 들어가지 않습니다.
    전송 도중 끊겨서 같은 묶음을 다시 보내도 중복 없이 한 번만 저장됩니다.
"""
import argparse
import json
import logging
import logging.handlers
import os
import re
import signal
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
log = logging.getLogger("collector")


# ------------------------------------------------------------------ 설정
def load_env(path: Path):
    """.env 파일의 KEY=VALUE 를 환경변수로 (이미 있는 값은 덮어쓰지 않음)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip("'").strip('"')
        os.environ.setdefault(k.strip(), v)


def _subst(obj):
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), obj)
    if isinstance(obj, list):
        return [_subst(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _subst(v) for k, v in obj.items()}
    return obj


def load_config(path: Path) -> dict:
    load_env(path.parent / ".env")
    with open(path, encoding="utf-8") as f:
        return _subst(yaml.safe_load(f))


# ------------------------------------------------------------------ 토픽 / 경로
def topic_match(pattern: str, topic: str) -> bool:
    p, t = pattern.split("/"), topic.split("/")
    for i, seg in enumerate(p):
        if seg == "#":
            return True
        if i >= len(t):
            return False
        if seg != "+" and seg != t[i]:
            return False
    return len(p) == len(t)


def get_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
        if cur is None:
            return None
    return cur


def resolve(spec, msg, topic_parts, key=None):
    """규칙의 값 지정(spec)을 실제 값으로. 목록이면 처음으로 값이 있는 것."""
    if spec is None:
        return None
    if isinstance(spec, list):
        for s in spec:
            v = resolve(s, msg, topic_parts, key)
            if v not in (None, ""):
                return v
        return None
    s = str(spec)
    if s.startswith("="):
        return s[1:]
    if s == "$key":
        return key
    if s == "topic":
        return "/".join(topic_parts)
    if s.startswith("topic:"):
        i = int(s[6:])
        return topic_parts[i] if i < len(topic_parts) else None
    return get_path(msg, s)


_ISO = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:?\d\d)?$")


def parse_time(val, kind: str):
    """→ epoch 초(float, UTC). 해석 못 하면 None."""
    if val is None:
        return None
    try:
        if kind == "epoch_ms":
            return float(val) / 1000.0
        if kind == "epoch_s":
            return float(val)
        if kind == "iso":
            m = _ISO.match(str(val).strip())
            if not m:
                return None
            base, frac, tz = m.groups()
            tz = tz or "+00:00"
            if tz == "Z":
                tz = "+00:00"
            if len(tz) == 5:  # +0900
                tz = tz[:3] + ":" + tz[3:]
            dt = datetime.fromisoformat(base + tz)
            return dt.timestamp() + (float("0." + frac) if frac else 0.0)
    except (ValueError, TypeError):
        return None
    return None


def to_number(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


# ------------------------------------------------------------------ 파서
class Parser:
    def __init__(self, rule: dict):
        self.name = rule["name"]
        self.topic = rule["topic"]
        self.rule = rule
        t = rule.get("time") or {"kind": "received"}
        self.time_path = t.get("path")
        self.time_kind = t.get("kind", "received")
        self.exclude = set(rule.get("exclude") or [])

    def parse(self, topic: str, payload: bytes, received_at: float):
        """→ (dedup_key, rows). rows = [(ts, space, point, device, metric, value)]"""
        msg = json.loads(payload.decode("utf-8"))
        parts = topic.split("/")
        r = self.rule

        ts = None
        if self.time_kind != "received":
            ts = parse_time(resolve(self.time_path, msg, parts), self.time_kind)
        if ts is None:
            ts = received_at

        dedup_key = None
        if r.get("dedup"):
            vals = [resolve(d, msg, parts) for d in r["dedup"]]
            if all(v is not None for v in vals):
                day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%d")
                dedup_key = f"{self.name}:" + ":".join(str(v) for v in vals) + f":{day}"

        groups = []  # (key, values_dict)
        if r.get("expand"):
            obj = resolve(r["expand"], msg, parts)
            if isinstance(obj, dict):
                groups = [(k, v) for k, v in obj.items() if isinstance(v, dict)]
        else:
            obj = resolve(r.get("values"), msg, parts) if r.get("values") else msg
            if isinstance(obj, dict):
                groups = [(None, obj)]

        rows = []
        for key, values in groups:
            space = resolve(r.get("space"), msg, parts, key) or "unknown"
            point = resolve(r.get("point"), msg, parts, key) or "-"
            device = resolve(r.get("device"), msg, parts, key) or "-"
            for metric, raw in values.items():
                if metric in self.exclude:
                    continue
                v = to_number(raw)
                if v is None:
                    continue
                rows.append((ts, str(space), str(point), str(device), str(metric), v))
        return dedup_key, rows


# ------------------------------------------------------------------ 저장소
SCHEMA = """
CREATE TABLE IF NOT EXISTS raw (
    id INTEGER PRIMARY KEY,
    received_at REAL NOT NULL,
    topic TEXT NOT NULL,
    payload BLOB
);
CREATE INDEX IF NOT EXISTS raw_recv ON raw(received_at);

CREATE TABLE IF NOT EXISTS messages (          -- 메시지 단위 중복 판별
    dedup_key TEXT PRIMARY KEY,
    first_received REAL,
    times_seen INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS duplicates (        -- 재전송으로 판정되어 버린 메시지 기록
    received_at REAL, topic TEXT, dedup_key TEXT
);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,              -- 메시지 안의 시각 (epoch 초, UTC)
    received_at REAL NOT NULL,     -- 수집기가 받은 시각
    space TEXT, point TEXT, device TEXT, metric TEXT,
    value REAL,
    parser TEXT,
    sent INTEGER DEFAULT 0         -- InfluxDB 전송 여부
);
CREATE INDEX IF NOT EXISTS m_series ON measurements(space, metric, ts);
CREATE INDEX IF NOT EXISTS m_unsent ON measurements(sent, id);
CREATE INDEX IF NOT EXISTS m_ts ON measurements(ts);

CREATE TABLE IF NOT EXISTS events (            -- 연결 끊김/재연결 등 (결측 원인 설명용)
    at REAL, kind TEXT, detail TEXT
);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")      # 읽는 쪽(현황·학습)이 쓰기를 막지 않게
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    # 이전 버전 DB 에 PostgreSQL 전송 여부 칸 추가 (이미 쌓인 데이터도 전송 대상이 됨)
    cols = {r[1] for r in con.execute("PRAGMA table_info(measurements)")}
    if "pg_sent" not in cols:
        con.execute("ALTER TABLE measurements ADD COLUMN pg_sent INTEGER DEFAULT 0")
    con.execute("CREATE INDEX IF NOT EXISTS m_pg_unsent ON measurements(pg_sent, id)")
    con.commit()
    return con


class Store:
    def __init__(self, con: sqlite3.Connection, keep_raw: bool):
        self.con = con
        self.keep_raw = keep_raw
        self.lock = threading.Lock()

    def event(self, kind, detail=""):
        with self.lock:
            self.con.execute("INSERT INTO events VALUES (?,?,?)", (time.time(), kind, detail))
            self.con.commit()

    def save(self, topic, payload, received_at, parser, dedup_key, rows, store_raw=True):
        """→ 'new' | 'dup' | 'empty'"""
        with self.lock:
            c = self.con
            if store_raw and self.keep_raw:
                c.execute("INSERT INTO raw(received_at, topic, payload) VALUES (?,?,?)",
                          (received_at, topic, payload))
            status = "empty"
            if dedup_key:
                cur = c.execute("INSERT OR IGNORE INTO messages(dedup_key, first_received) VALUES (?,?)",
                                (dedup_key, received_at))
                if cur.rowcount == 0:
                    c.execute("UPDATE messages SET times_seen = times_seen + 1 WHERE dedup_key=?", (dedup_key,))
                    c.execute("INSERT INTO duplicates VALUES (?,?,?)", (received_at, topic, dedup_key))
                    c.commit()
                    return "dup"
            if rows:
                c.executemany(
                    "INSERT INTO measurements(ts, received_at, space, point, device, metric, value, parser)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    [(r[0], received_at, r[1], r[2], r[3], r[4], r[5], parser) for r in rows])
                status = "new"
            c.commit()
            return status


# ------------------------------------------------------------------ 처리 파이프라인
class Pipeline:
    def __init__(self, cfg, store: Store):
        self.store = store
        self.parsers = [Parser(r) for r in cfg.get("parsers", [])]
        self.ignore = cfg.get("ignore", []) or []
        self.stats = {"received": 0, "new": 0, "dup": 0, "unmatched": 0, "error": 0, "ignored": 0}
        self.unmatched_seen = set()

    def handle(self, topic: str, payload: bytes, received_at: float, store_raw=True):
        self.stats["received"] += 1
        if any(topic_match(p, topic) for p in self.ignore):
            self.stats["ignored"] += 1
            return
        parser = next((p for p in self.parsers if topic_match(p.topic, topic)), None)
        if parser is None:
            self.stats["unmatched"] += 1
            pattern = re.sub(r"[0-9a-fA-F-]{8,}", "*", topic)
            if pattern not in self.unmatched_seen:
                self.unmatched_seen.add(pattern)
                log.warning("규칙 없는 토픽(원본만 저장): %s", topic)
            if store_raw and self.store.keep_raw:
                self.store.save(topic, payload, received_at, None, None, [], store_raw)
            return
        try:
            key, rows = parser.parse(topic, payload, received_at)
        except Exception as e:  # 형식이 바뀐 메시지 등 - 원본은 남기고 계속 진행
            self.stats["error"] += 1
            log.warning("파싱 실패 [%s] %s: %s", parser.name, topic, e)
            self.store.save(topic, payload, received_at, parser.name, None, [], store_raw)
            return
        result = self.store.save(topic, payload, received_at, parser.name, key, rows, store_raw)
        if result in self.stats:
            self.stats[result] += 1


# ------------------------------------------------------------------ InfluxDB 전송
def _esc_tag(s):
    return str(s).replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


class InfluxSender(threading.Thread):
    """measurements 의 sent=0 행을 모아서 InfluxDB 로 보냄. 실패하면 기다렸다가 다시."""

    def __init__(self, cfg, db_path: Path, stop: threading.Event):
        super().__init__(daemon=True, name="influx")
        self.cfg = cfg
        self.con = open_db(db_path)
        self.stop = stop
        q = urllib.parse.urlencode({"org": cfg["org"], "bucket": cfg["bucket"], "precision": "ms"})
        self.url = cfg["url"].rstrip("/") + "/api/v2/write?" + q
        self.meas = _esc_tag(cfg.get("measurement", "sensor"))
        self.pending = 0

    def run(self):
        backoff = 1
        while not self.stop.is_set():
            try:
                sent = self.flush_once()
                backoff = 1
                if sent == 0:
                    self.stop.wait(self.cfg.get("flush_interval_sec", 5))
            except Exception as e:
                log.warning("InfluxDB 전송 실패 (%ss 후 재시도): %s", backoff, e)
                self.stop.wait(backoff)
                backoff = min(backoff * 2, 60)

    def flush_once(self) -> int:
        rows = self.con.execute(
            "SELECT id, ts, space, point, device, metric, value FROM measurements"
            " WHERE sent=0 ORDER BY id LIMIT ?", (self.cfg.get("batch_size", 1000),)).fetchall()
        self.pending = self.con.execute("SELECT COUNT(*) FROM measurements WHERE sent=0").fetchone()[0]
        if not rows:
            return 0
        lines = []
        for _id, ts, space, point, device, metric, value in rows:
            lines.append(f"{self.meas},space={_esc_tag(space)},point={_esc_tag(point)},device={_esc_tag(device)} "
                         f"{_esc_tag(metric)}={value} {int(round(ts * 1000))}")
        req = urllib.request.Request(self.url, data="\n".join(lines).encode("utf-8"), method="POST",
                                     headers={"Authorization": f"Token {self.cfg['token']}",
                                              "Content-Type": "text/plain; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"HTTP {resp.status}")
        ids = [r[0] for r in rows]
        self.con.executemany("UPDATE measurements SET sent=1 WHERE id=?", [(i,) for i in ids])
        self.con.commit()
        return len(rows)


# ------------------------------------------------------------------ PostgreSQL 전송
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def pg_ddl(schema: str, table: str) -> str:
    """PostgreSQL 에 만들 테이블. 공간·항목·시각으로 찾기 쉽게 인덱스를 둠."""
    t = f"{schema}.{table}"
    return f"""
CREATE TABLE IF NOT EXISTS {t} (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ      NOT NULL,   -- 메시지 안의 시각 (센서 측정 시각)
    received_at  TIMESTAMPTZ      NOT NULL,   -- 수집기가 받은 시각
    space        TEXT             NOT NULL,   -- 공간 (실습실, 개발실-실내기 ...)
    point        TEXT             NOT NULL,   -- 세부 위치/구분 (idu, odu ...)
    device       TEXT             NOT NULL,   -- 기기 이름
    metric       TEXT             NOT NULL,   -- 항목 (temperature, co2 ...)
    value        DOUBLE PRECISION NOT NULL,
    parser       TEXT,                        -- 적용한 파싱 규칙 (lora, hvac, controller)
    inserted_at  TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT {table}_uniq UNIQUE (space, point, device, metric, ts)
);
CREATE INDEX IF NOT EXISTS {table}_series ON {t} (space, metric, ts);
CREATE INDEX IF NOT EXISTS {table}_ts     ON {t} (ts);
"""


class PgSender(threading.Thread):
    """measurements 의 pg_sent=0 행을 모아서 PostgreSQL 로 보냄. 실패하면 다시 접속해서 재시도."""

    def __init__(self, cfg, db_path: Path, stop: threading.Event):
        super().__init__(daemon=True, name="postgres")
        self.cfg = cfg
        self.con = open_db(db_path)
        self.stop = stop
        self.schema = cfg.get("schema") or cfg["user"]
        self.table = cfg.get("table", "measurements")
        for name in (self.schema, self.table):
            if not _IDENT.match(name):
                raise ValueError(f"schema/table 이름은 영문·숫자·밑줄만: {name!r}")
        self.pg = None
        self.pending = 0

    def connect(self):
        import psycopg2
        self.pg = psycopg2.connect(
            host=self.cfg["host"], port=int(self.cfg.get("port", 5432)),
            dbname=self.cfg["dbname"], user=self.cfg["user"], password=self.cfg.get("password"),
            connect_timeout=int(self.cfg.get("connect_timeout", 10)),
            application_name="sensor-collector",
            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=3)
        with self.pg.cursor() as cur:
            cur.execute(pg_ddl(self.schema, self.table))
        self.pg.commit()
        log.info("PostgreSQL 접속 성공: %s:%s/%s (%s.%s)", self.cfg["host"], self.cfg.get("port"),
                 self.cfg["dbname"], self.schema, self.table)

    def close(self):
        try:
            if self.pg is not None:
                self.pg.close()
        except Exception:
            pass
        self.pg = None

    def run(self):
        backoff = 1
        while not self.stop.is_set():
            try:
                if self.pg is None or self.pg.closed:
                    self.connect()
                sent = self.flush_once()
                backoff = 1
                if sent == 0:
                    self.stop.wait(self.cfg.get("flush_interval_sec", 5))
            except Exception as e:
                log.warning("PostgreSQL 전송 실패 (%ss 후 재시도): %s", backoff, str(e).strip())
                self.close()
                self.stop.wait(backoff)
                backoff = min(backoff * 2, 60)
        self.close()

    def flush_once(self) -> int:
        from psycopg2.extras import execute_values
        rows = self.con.execute(
            "SELECT id, ts, received_at, space, point, device, metric, value, parser FROM measurements"
            " WHERE pg_sent=0 ORDER BY id LIMIT ?", (self.cfg.get("batch_size", 1000),)).fetchall()
        self.pending = self.con.execute("SELECT COUNT(*) FROM measurements WHERE pg_sent=0").fetchone()[0]
        if not rows:
            return 0
        data = [(datetime.fromtimestamp(ts, timezone.utc), datetime.fromtimestamp(rcv, timezone.utc),
                 space, point, device, metric, value, parser)
                for _id, ts, rcv, space, point, device, metric, value, parser in rows]
        with self.pg.cursor() as cur:
            execute_values(
                cur,
                f"INSERT INTO {self.schema}.{self.table}"
                " (ts, received_at, space, point, device, metric, value, parser) VALUES %s"
                " ON CONFLICT (space, point, device, metric, ts) DO NOTHING",
                data, page_size=500)
        self.pg.commit()   # 커밋이 끝난 뒤에만 '보냄'으로 표시 → 도중에 끊겨도 잃지 않음
        self.con.executemany("UPDATE measurements SET pg_sent=1 WHERE id=?", [(r[0],) for r in rows])
        self.con.commit()
        return len(rows)


# ------------------------------------------------------------------ 실행 모드
def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h1 = logging.StreamHandler(sys.stdout)
    h1.setFormatter(fmt)
    h2 = logging.handlers.RotatingFileHandler(log_dir / "collector.log", maxBytes=5_000_000,
                                              backupCount=3, encoding="utf-8")
    h2.setFormatter(fmt)
    log.addHandler(h1)
    log.addHandler(h2)
    log.setLevel(logging.INFO)


def run_mqtt(cfg, pipe: Pipeline, store: Store, influx: "InfluxSender|None", stop: threading.Event,
             pg: "PgSender|None" = None):
    import paho.mqtt.client as mqtt

    m = cfg["mqtt"]
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=m.get("client_id") or "",
                         clean_session=False)   # 끊긴 동안의 QoS1 메시지를 브로커가 보관해 줌
    if m.get("username"):
        client.username_pw_set(m["username"], m.get("password"))
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def on_connect(cl, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("브로커 접속 거부: %s", reason_code)
            store.event("connect_fail", str(reason_code))
            return
        for t in m.get("topics", ["#"]):
            cl.subscribe(t, qos=m.get("qos", 1))   # 재연결할 때마다 다시 구독
        log.info("브로커 접속 성공 %s:%s, 구독 %s", m["host"], m["port"], m.get("topics"))
        store.event("connected", f"{m['host']}:{m['port']}")

    def on_disconnect(cl, userdata, flags, reason_code, properties):
        log.warning("브로커 연결 끊김: %s (자동 재연결 시도)", reason_code)
        store.event("disconnected", str(reason_code))

    def on_message(cl, userdata, msg):
        pipe.handle(msg.topic, msg.payload, time.time())

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    def heartbeat():
        last = dict(pipe.stats)
        while not stop.wait(60):
            s = pipe.stats
            d = {k: s[k] - last.get(k, 0) for k in s}
            last = dict(s)
            extra = (f", PostgreSQL 대기 {pg.pending}" if pg else "") + \
                    (f", InfluxDB 대기 {influx.pending}" if influx else "")
            log.info("최근 1분: 수신 %d, 저장 %d, 중복 %d, 규칙없음 %d, 오류 %d%s",
                     d["received"], d["new"], d["dup"], d["unmatched"], d["error"], extra)
            if d["received"] == 0:
                log.warning("1분 동안 수신 0건 — 네트워크/브로커 확인 필요")
                store.event("no_data_1min", "")

    threading.Thread(target=heartbeat, daemon=True).start()

    def shutdown(*_):
        log.info("종료 신호 받음")
        stop.set()
        client.disconnect()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    store.event("start", "")
    client.connect_async(m["host"], int(m.get("port", 1883)), keepalive=int(m.get("keepalive", 30)))
    client.loop_forever(retry_first_connection=True)   # 시작할 때 네트워크가 없어도 계속 재시도
    store.event("stop", "")


def run_replay(path: Path, pipe: Pipeline):
    """mosquitto_sub -v 출력 파일('토픽 내용' 한 줄씩)을 재생."""
    n = 0
    with open(path, "rb") as f:
        for line in f:
            line = line.rstrip(b"\r\n")
            if b" " not in line:
                continue
            topic, payload = line.split(b" ", 1)
            pipe.handle(topic.decode("utf-8", "replace"), payload, time.time())
            n += 1
    log.info("재생 완료: %d줄, %s", n, pipe.stats)


def run_reprocess(con: sqlite3.Connection, pipe: Pipeline):
    """원본(raw)을 현재 규칙으로 다시 처리. 기존 정리 결과는 지우고 새로 만듦."""
    log.info("기존 측정값/중복 기록을 지우고 원본에서 다시 만듭니다")
    with pipe.store.lock:
        con.executescript("DELETE FROM measurements; DELETE FROM messages; DELETE FROM duplicates;")
        con.commit()
    for topic, payload, received_at in con.execute(
            "SELECT topic, payload, received_at FROM raw ORDER BY id").fetchall():
        pipe.handle(topic, payload, received_at, store_raw=False)
    log.info("다시 처리 완료: %s", pipe.stats)


def main():
    ap = argparse.ArgumentParser(description="MQTT 센서 데이터 수집기")
    ap.add_argument("-c", "--config", default=str(BASE / "config.yaml"))
    ap.add_argument("--replay", help="mosquitto_sub -v 로 저장한 파일을 재생")
    ap.add_argument("--reprocess", action="store_true", help="저장된 원본을 현재 규칙으로 다시 처리")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    os.chdir(cfg_path.parent)
    db_path = Path(cfg["storage"]["sqlite_path"]).resolve()
    setup_logging(db_path.parent)

    con = open_db(db_path)
    store = Store(con, cfg["storage"].get("keep_raw", True))
    pipe = Pipeline(cfg, store)
    log.info("규칙 %d개 로드: %s", len(pipe.parsers), [p.name for p in pipe.parsers])

    if args.replay:
        run_replay(Path(args.replay), pipe)
        return
    if args.reprocess:
        run_reprocess(con, pipe)
        return

    stop = threading.Event()
    influx = None
    ic = cfg.get("influx") or {}
    if ic.get("enabled"):
        if not (ic.get("org") and ic.get("bucket") and ic.get("token")):
            log.error("influx.enabled=true 인데 org/bucket/token 중 빈 값이 있습니다. InfluxDB 전송은 끕니다.")
        else:
            influx = InfluxSender(ic, db_path, stop)
            influx.start()
            log.info("InfluxDB 전송 켜짐: %s (org=%s, bucket=%s)", ic["url"], ic["org"], ic["bucket"])
    else:
        log.info("InfluxDB 전송 꺼짐")

    pg = None
    pc = cfg.get("postgres") or {}
    if pc.get("enabled"):
        missing = [k for k in ("host", "dbname", "user", "password") if not pc.get(k)]
        if missing:
            log.error("postgres.enabled=true 인데 빈 값이 있습니다: %s. PostgreSQL 전송은 끕니다.", missing)
        else:
            pg = PgSender(pc, db_path, stop)
            pg.start()
            log.info("PostgreSQL 전송 켜짐: %s:%s/%s (스키마 %s)", pc["host"], pc.get("port"),
                     pc["dbname"], pg.schema)
    else:
        log.info("PostgreSQL 전송 꺼짐 (로컬에만 저장, 나중에 켜면 밀린 데이터부터 전송)")

    run_mqtt(cfg, pipe, store, influx, stop, pg)


if __name__ == "__main__":
    main()
