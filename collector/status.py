#!/usr/bin/env python3
"""
수집 현황 보기 — "지금 데이터가 잘 쌓이고 있는가?"

  python3 status.py                 # 최근 24시간, 젯슨 안 SQLite 기준
  python3 status.py --source pg     # 최종 저장소(PostgreSQL) 기준
  python3 status.py --hours 1       # 최근 1시간
  python3 status.py --gap-min 5     # 5분 이상 빈 구간만 결측으로 표시
  python3 status.py --out report.md # 결과를 파일로도 저장 (보고서용)

출력
  1) 공간·항목별: 기기 수, 건수, 마지막 수신, 전송 주기, 지연, 커버율
  2) 결측 구간과 추정 원인
  3) 수집기 이벤트(끊김/재연결), 중복 메시지 수, SQLite ↔ PostgreSQL 전송 상태

커버율 정의
  확인 구간 안의 '다 지나간 1분 칸' 중 데이터가 1건 이상 있는 칸의 비율.
  - 구간의 시작·끝에 걸친 덜 찬 1분 칸과 아직 진행 중인 현재 1분 칸은 세지 않습니다.
  - 센서 시계가 빨라 구간 밖 시각으로 찍힌 데이터는 칸 수에 넣지 않습니다.
  그래서 커버율은 항상 0~100% 사이입니다.
"""
import argparse
import math
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from collector import load_config  # noqa: E402  (.env 를 읽어 ${PG_PASSWORD} 등을 채움)


def fmt_t(t):
    return datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S") if t else "-"


def fmt_ago(sec):
    if sec < 90:
        return f"{max(sec, 0):.0f}초 전"
    if sec < 5400:
        return f"{sec / 60:.0f}분 전"
    return f"{sec / 3600:.1f}시간 전"


# ------------------------------------------------------------------ 데이터 읽기
class SqliteSource:
    name = "SQLite (젯슨 안)"

    def __init__(self, con):
        self.con = con

    def first_last(self):
        return self.con.execute("SELECT MIN(ts), MAX(ts) FROM measurements").fetchone()

    def rows(self, start):
        return self.con.execute(
            "SELECT space, metric, device, ts, received_at FROM measurements WHERE ts >= ? ORDER BY ts",
            (start,)).fetchall()


class PgSource:
    def __init__(self, pc):
        import psycopg2
        self.pg = psycopg2.connect(
            host=pc["host"], port=int(pc.get("port", 5432)), dbname=pc["dbname"],
            user=pc["user"], password=pc.get("password"), connect_timeout=10,
            application_name="sensor-status")
        self.pg.set_session(readonly=True, autocommit=True)
        schema = pc.get("schema") or pc["user"]
        self.table = f"{schema}.{pc.get('table', 'measurements')}"
        self.name = f"PostgreSQL ({pc['host']}:{pc.get('port')}/{pc['dbname']}, {self.table})"

    def _q(self, sql, args=()):
        with self.pg.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    def first_last(self):
        r = self._q(f"SELECT EXTRACT(EPOCH FROM MIN(ts)), EXTRACT(EPOCH FROM MAX(ts)) FROM {self.table}")[0]
        return tuple(float(x) if x is not None else None for x in r)

    def rows(self, start):
        return [(s, m, d, float(t), float(r)) for s, m, d, t, r in self._q(
            f"SELECT space, metric, device, EXTRACT(EPOCH FROM ts), EXTRACT(EPOCH FROM received_at)"
            f" FROM {self.table} WHERE ts >= to_timestamp(%s) ORDER BY ts", (start,))]

    def count(self):
        return int(self._q(f"SELECT COUNT(*) FROM {self.table}")[0][0])


# ------------------------------------------------------------------ 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=str(BASE / "config.yaml"))
    ap.add_argument("--source", choices=["sqlite", "pg"], default="sqlite",
                    help="sqlite: 젯슨 안 저장소 / pg: PostgreSQL")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--gap-min", type=int, default=3, help="이 분 수 이상 연속으로 비면 결측으로 표시")
    ap.add_argument("--out", help="결과를 마크다운 파일로도 저장")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    db = (cfg_path.parent / cfg["storage"]["sqlite_path"]).resolve()
    lite = sqlite3.connect(f"file:{db}?mode=ro", uri=True) if db.exists() else None  # 읽기 전용

    if args.source == "pg":
        pc = cfg.get("postgres") or {}
        if not pc.get("password"):
            sys.exit("config.yaml 의 postgres 설정 또는 .env 의 PG_PASSWORD 가 비어 있습니다.")
        try:
            src = PgSource(pc)
        except Exception as e:
            sys.exit(f"PostgreSQL 접속 실패: {str(e).strip()}")
    else:
        if lite is None:
            sys.exit(f"DB 파일이 없습니다: {db}  (수집기를 먼저 실행하세요)")
        src = SqliteSource(lite)

    now = time.time()
    first, last_any = src.first_last()
    if first is None:
        sys.exit("아직 저장된 측정값이 없습니다.")

    # 확인 구간: [start, now].  커버율은 이 안에 '온전히 들어가고 다 지나간' 1분 칸만 셈
    start = max(now - args.hours * 3600, first)
    m0 = math.ceil(start / 60)          # 첫 온전한 1분 칸
    m1 = int(now // 60)                 # 현재(진행 중) 1분 칸 → 제외
    slots = range(m0, m1)               # m0 ~ m1-1
    total_min = max(len(slots), 1)

    rows = src.rows(start)

    series = defaultdict(lambda: {"devices": set(), "n": 0, "last_recv": 0,
                                  "minutes": set(), "lags": [], "dev_ts": defaultdict(list)})
    all_minutes = set()
    for space, metric, device, ts, recv in rows:
        s = series[(space, metric)]
        s["devices"].add(device)
        s["n"] += 1
        s["last_recv"] = max(s["last_recv"], recv)
        mi = int(ts // 60)
        if m0 <= mi < m1:               # 구간 밖(덜 찬 칸, 미래 시각)은 커버율 계산에서 뺌
            s["minutes"].add(mi)
            all_minutes.add(mi)
        s["lags"].append(recv - ts)
        s["dev_ts"][device].append(ts)

    if lite is not None:
        events = lite.execute("SELECT at, kind, detail FROM events WHERE at >= ? ORDER BY at", (start,)).fetchall()
        dups = lite.execute("SELECT COUNT(*) FROM duplicates WHERE received_at >= ?", (start,)).fetchone()[0]
    else:
        events, dups = [], 0
    down = [int(a // 60) for a, k, _ in events if k in ("disconnected", "no_data_1min", "stop", "connect_fail", "start")]

    out = []
    p = out.append
    p(f"# 수집 현황  ({fmt_t(start)} ~ {fmt_t(now)})\n")
    p(f"- 기준 저장소: {src.name}")
    p(f"- 측정값 {len(rows):,}건, 공간·항목 {len(series)}개, 재전송(중복)으로 버린 메시지 {dups}건")
    p(f"- 커버율 계산 칸: {fmt_t(m0 * 60)} ~ {fmt_t(m1 * 60)}, 다 지나간 1분 칸 {total_min}개")
    p(f"- 전체 커버율(어느 센서든 데이터가 있는 칸): {len(all_minutes) / total_min * 100:.1f}%")
    if lite is not None:
        cols = {r[1] for r in lite.execute("PRAGMA table_info(measurements)")}
        if "pg_sent" in cols:
            n_all, n_wait = lite.execute(
                "SELECT COUNT(*), COALESCE(SUM(pg_sent=0),0) FROM measurements").fetchone()
            line = f"- SQLite 전체 {n_all:,}건 중 PostgreSQL 전송 대기 {n_wait:,}건"
            if isinstance(src, PgSource):
                line += f" / PostgreSQL 전체 {src.count():,}건"
            p(line)
    p("")

    p("## 공간·항목별\n")
    p("| 공간 | 항목 | 기기 | 건수 | 마지막 수신 | 주기 | 지연(중앙) | 커버율 | 판정 |")
    p("|---|---|---:|---:|---|---:|---:|---:|---|")
    gaps_out = []
    for (space, metric), s in sorted(series.items()):
        periods = []
        for tss in s["dev_ts"].values():
            d = [b - a for a, b in zip(tss, tss[1:]) if b > a]
            if d:
                periods.append(statistics.median(d))
        period = statistics.median(periods) if periods else None
        cov = len(s["minutes"]) / total_min * 100
        lag = statistics.median(s["lags"]) if s["lags"] else 0
        age = now - s["last_recv"]
        if age > max(600, (period or 60) * 5):
            verdict = "⛔ 끊김"
        elif cov >= 95:
            verdict = "✅"
        elif period and period > 90:
            # 1분보다 느리게 보내는 센서는 원래 95%가 나올 수 없음 → 주기 기준 기대치와 비교
            expected = min(100, 60 / period * 100 * len(s["devices"]))
            verdict = "✅ (주기상 정상)" if cov >= expected * 0.9 else f"⚠ 주기 {period:.0f}s, 기대 ≈{expected:.0f}%"
        else:
            verdict = "⚠ 확인"
        p(f"| {space} | {metric} | {len(s['devices'])} | {s['n']:,} | {fmt_ago(age)} | "
          f"{(f'{period:.0f}s') if period else '-'} | {lag:.1f}s | {cov:.1f}% | {verdict} |")

        # 결측 구간 (커버율과 같은 칸 기준)
        missing, run_start = [], None
        for mi in slots:
            if mi not in s["minutes"]:
                run_start = mi if run_start is None else run_start
            elif run_start is not None:
                missing.append((run_start, mi))
                run_start = None
        if run_start is not None:
            missing.append((run_start, m1))
        for a, b in missing:
            length = b - a
            if length < args.gap_min:
                continue
            if period and length * 60 < period * 2.5:
                continue   # 전송 주기 때문에 생긴 정상적인 빈칸
            if any(a - 1 <= dm <= b for dm in down):
                cause = "수집기 끊김/재시작 기록 있음 (네트워크·브로커·보드)"
            elif all(mi not in all_minutes for mi in range(a, b)):
                cause = "모든 센서 동시 결측 → 수집기·네트워크·보드 전원 쪽 추정"
            else:
                cause = "이 항목만 결측 → 센서/게이트웨이 쪽 추정"
            gaps_out.append((space, metric, a, b, length, cause))

    p("\n## 결측 구간 (추정 원인)\n")
    if gaps_out:
        # 같은 시각에 같이 빈 것들은 한 줄로 묶음 (공간별로 항목 나열)
        grouped = defaultdict(lambda: defaultdict(list))
        for space, metric, a, b, length, cause in gaps_out:
            grouped[(a, b, length, cause)][space].append(metric)
        p("| 시작 | 끝 | 길이 | 공간 (항목) | 추정 원인 |")
        p("|---|---|---:|---|---|")
        for (a, b, length, cause), spaces in sorted(grouped.items()):
            where = ", ".join(f"{sp}({len(ms)}개)" if len(ms) > 3 else f"{sp}({'/'.join(ms)})"
                              for sp, ms in sorted(spaces.items()))
            p(f"| {fmt_t(a * 60)} | {fmt_t(b * 60)} | {length}분 | {where} | {cause} |")
    else:
        p(f"{args.gap_min}분 이상 이어진 결측 없음")

    p("\n## 수집기 이벤트\n")
    if events:
        for a, k, d in events[-30:]:
            p(f"- {fmt_t(a)}  {k}  {d}")
    else:
        p("- 없음")

    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n→ {args.out} 에 저장")


if __name__ == "__main__":
    main()
