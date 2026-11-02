#!/usr/bin/env python3
"""
보관 정책 실행기 (다운샘플링 + 오래된 데이터 정리) — cron 으로 하루 한 번 실행

  venv/bin/python downsample.py            # 실행
  venv/bin/python downsample.py --dry-run  # 무엇을 할지 건수만 보여주고 바꾸지 않음

정책 (config.yaml 의 retention: 으로 바꿀 수 있음, 없으면 아래 기본값)
  PostgreSQL
    measurements      원본          raw_days (30일) 지나면 삭제 — 단, 집계가 끝난 구간만
    measurements_10m  10분 집계     agg10_days (365일) 지나면 삭제
    measurements_1h   1시간 집계    계속 보관
  Jetson SQLite (data/collector.db)
    raw               원본 메시지   sqlite_raw_days (14일) 지나면 삭제
    measurements      PostgreSQL 로 전송 끝난 행만 sqlite_sent_days (7일) 지나면 삭제
    duplicates        sqlite_raw_days 지나면 삭제

집계 값: 평균·최소·최대·건수. 최소·최대는 CO2 급상승 같은 순간값을, 건수는 나중에 커버율을 계산할 수 있게 남김.
여러 번 실행해도 결과가 같음(같은 구간은 다시 계산해 덮어씀). cron 이 며칠 멈췄다가 돌아도 밀린 구간부터 처리함.
"""
import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from collector import load_config  # noqa: E402

log = logging.getLogger("downsample")

DEFAULTS = {"raw_days": 30, "agg10_days": 365, "sqlite_raw_days": 14, "sqlite_sent_days": 7}
LEVELS = [("measurements_10m", 600), ("measurements_1h", 3600)]


def agg_ddl(schema, table):
    return f"""
CREATE TABLE IF NOT EXISTS {schema}.{table} (
    bucket  TIMESTAMPTZ      NOT NULL,   -- 구간 시작 시각
    space   TEXT             NOT NULL,
    metric  TEXT             NOT NULL,
    avg_v   DOUBLE PRECISION NOT NULL,
    min_v   DOUBLE PRECISION NOT NULL,
    max_v   DOUBLE PRECISION NOT NULL,
    n       INTEGER          NOT NULL,   -- 구간 안 측정값 수
    PRIMARY KEY (bucket, space, metric)
);
CREATE INDEX IF NOT EXISTS {table}_series ON {schema}.{table} (space, metric, bucket);
"""


def one(cur, sql, args=()):
    cur.execute(sql, args)
    return cur.fetchall()[0][0]


def run_pg(pc, ret, dry):
    import psycopg2
    con = psycopg2.connect(host=pc["host"], port=int(pc.get("port", 5432)), dbname=pc["dbname"],
                           user=pc["user"], password=pc.get("password"), connect_timeout=15,
                           application_name="sensor-downsample")
    schema = pc.get("schema") or pc["user"]
    raw = f"{schema}.{pc.get('table', 'measurements')}"
    cur = con.cursor()
    for table, _ in LEVELS:
        cur.execute(agg_ddl(schema, table))
    con.commit()

    summary = {}
    for table, sec in LEVELS:
        t = f"{schema}.{table}"
        # 어디서부터 집계할지: 마지막 집계 구간 하루 전부터(늦게 도착한 데이터 반영), 처음이면 원본의 맨 처음부터
        # 어디까지: 지금 진행 중인 구간은 빼고 다 끝난 구간까지
        bucket = f"to_timestamp(floor(extract(epoch from ts) / {sec}) * {sec})"
        rng = (f"ts >= COALESCE((SELECT max(bucket) FROM {t}) - interval '1 day', '-infinity')"
               f" AND ts < to_timestamp(floor(extract(epoch from now()) / {sec}) * {sec})")
        if dry:
            n = one(cur, f"SELECT count(DISTINCT ({bucket}, space, metric)) FROM {raw} WHERE {rng}")
        else:
            n = one(cur, f"""
WITH up AS (
  INSERT INTO {t} (bucket, space, metric, avg_v, min_v, max_v, n)
  SELECT {bucket}, space, metric, avg(value), min(value), max(value), count(*)
  FROM {raw} WHERE {rng}
  GROUP BY 1, 2, 3
  ON CONFLICT (bucket, space, metric) DO UPDATE
    SET avg_v = EXCLUDED.avg_v, min_v = EXCLUDED.min_v, max_v = EXCLUDED.max_v, n = EXCLUDED.n
  RETURNING 1)
SELECT count(*) FROM up""")
        summary[table] = n
        log.info("%s 집계 %s구간%s", table, f"{n:,}", " (dry-run)" if dry else "")

    # 원본 삭제: 보관 기간이 지났고, 10분·1시간 집계가 모두 끝난 구간만
    cond = (f"ts < now() - interval '{int(ret['raw_days'])} days'"
            + "".join(f" AND ts < (SELECT COALESCE(max(bucket), '-infinity') FROM {schema}.{t})" for t, _ in LEVELS))
    agg10 = f"bucket < now() - interval '{int(ret['agg10_days'])} days'"
    if dry:
        d_raw = one(cur, f"SELECT count(*) FROM {raw} WHERE {cond}")
        d_10 = one(cur, f"SELECT count(*) FROM {schema}.measurements_10m WHERE {agg10}")
    else:
        d_raw = one(cur, f"WITH d AS (DELETE FROM {raw} WHERE {cond} RETURNING 1) SELECT count(*) FROM d")
        d_10 = one(cur, f"WITH d AS (DELETE FROM {schema}.measurements_10m WHERE {agg10} RETURNING 1) SELECT count(*) FROM d")
        con.commit()
    log.info("PostgreSQL 원본 %s일 지난 행 삭제 %s건, 10분 집계 %s일 지난 행 삭제 %s건%s",
             ret["raw_days"], f"{d_raw:,}", ret["agg10_days"], f"{d_10:,}", " (dry-run)" if dry else "")

    sizes = {}
    for t in [pc.get("table", "measurements")] + [t for t, _ in LEVELS]:
        sizes[t] = one(cur, f"SELECT pg_size_pretty(pg_total_relation_size('{schema}.{t}'))")
    log.info("PostgreSQL 크기: %s", ", ".join(f"{k} {v}" for k, v in sizes.items()))
    con.close()


def run_sqlite(db, ret, dry):
    if not db.exists():
        log.info("SQLite 파일 없음: %s (건너뜀)", db)
        return
    con = sqlite3.connect(str(db), timeout=60)
    now = time.time()
    raw_cut = now - float(ret["sqlite_raw_days"]) * 86400
    sent_cut = now - float(ret["sqlite_sent_days"]) * 86400
    cols = {r[1] for r in con.execute("PRAGMA table_info(measurements)")}
    sent_col = "pg_sent" if "pg_sent" in cols else None
    jobs = [("raw", "received_at < ?", raw_cut), ("duplicates", "received_at < ?", raw_cut)]
    if sent_col:   # PostgreSQL 로 아직 못 보낸 행은 절대 지우지 않음
        jobs.append(("measurements", f"{sent_col} = 1 AND received_at < ?", sent_cut))
    for table, where, cut in jobs:
        if dry:
            n = con.execute(f"SELECT count(*) FROM {table} WHERE {where}", (cut,)).fetchone()[0]
        else:
            n = con.execute(f"DELETE FROM {table} WHERE {where}", (cut,)).rowcount
            con.commit()
        log.info("SQLite %s 삭제 %s건%s", table, f"{n:,}", " (dry-run)" if dry else "")
    if not dry:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # WAL 파일이 계속 커지지 않게
    size = db.stat().st_size / 1e6
    log.info("SQLite 크기: %.1f MB (지운 공간은 새 데이터가 다시 씀)", size)
    con.close()


def main():
    ap = argparse.ArgumentParser(description="보관 정책 실행 (다운샘플링 + 정리)")
    ap.add_argument("-c", "--config", default=str(BASE / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="건수만 보여주고 바꾸지 않음")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    db = (cfg_path.parent / cfg["storage"]["sqlite_path"]).resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(db.parent / "downsample.log", encoding="utf-8")])
    ret = {**DEFAULTS, **(cfg.get("retention") or {})}
    log.info("시작 %s 정책 %s", "(dry-run)" if args.dry_run else "", ret)

    ok = True
    pc = cfg.get("postgres") or {}
    if pc.get("enabled") and pc.get("password"):
        try:
            run_pg(pc, ret, args.dry_run)
        except Exception as e:
            ok = False
            log.error("PostgreSQL 작업 실패 (다음 실행 때 밀린 구간부터 다시 함): %s", str(e).strip())
    else:
        log.info("PostgreSQL 설정이 꺼져 있어 건너뜀")
    try:
        run_sqlite(db, ret, args.dry_run)
    except Exception as e:
        ok = False
        log.error("SQLite 정리 실패: %s", e)
    log.info("끝 (%s)", "정상" if ok else "일부 실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
