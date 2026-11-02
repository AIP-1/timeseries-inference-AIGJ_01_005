#!/usr/bin/env python3
"""
과제 8 — 가동 로그 요약 (logs/cycles.jsonl)
  ~/collector/venv/bin/python log_summary.py                 # 마지막으로 서비스를 켠 뒤부터
  ~/collector/venv/bin/python log_summary.py --since "2026-10-21 15:30"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/cycles.jsonl")
    ap.add_argument("--since", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.log).read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.ts, format="ISO8601")
    df = df.sort_values("ts").reset_index(drop=True)
    gap = df.ts.diff().dt.total_seconds()
    if a.since:
        df = df[df.ts >= pd.Timestamp(a.since, tz=df.ts.dt.tz)]
    else:   # 마지막으로 5분 넘게 끊긴 뒤(=서비스 재시작)부터
        restarts = df.index[gap > 300]
        if len(restarts):
            df = df.loc[restarts[-1]:]
    df = df.reset_index(drop=True)
    gap = df.ts.diff().dt.total_seconds().fillna(60)
    span_h = (df.ts.iloc[-1] - df.ts.iloc[0]).total_seconds() / 3600
    expected = int(round(span_h * 60)) + 1
    miss = gap[gap > 90]

    print("== ① 가동 시간 ==")
    print(f"시작 {df.ts.iloc[0]:%m-%d %H:%M}  →  마지막 {df.ts.iloc[-1]:%m-%d %H:%M}  =  {span_h:.1f}시간")
    print(f"주기 {len(df):,}번 (매분이면 {expected:,}번) | 1분 넘게 빈 곳 {len(miss)}번" +
          (f", 가장 길게 빈 시간 {miss.max():.0f}초" if len(miss) else " → 무중단"))
    print(f"계산 오류가 난 주기 {int((df.errors.astype(str) != '{}').sum())}번 | MQTT 정상 발행 {df.mqtt_ok.mean()*100:.1f}%")

    print("\n== ② 주기별 처리 시간 (ms, 공간 3곳 전부) ==")
    ms = df.ms.astype(float)
    print(f"평균 {ms.mean():.0f} | 중간값 {ms.median():.0f} | 느린 5% {np.percentile(ms, 95):.0f} | 최대 {ms.max():.0f} "
          f"(최대는 {df.ts[ms.idxmax()]:%m-%d %H:%M})")
    hourly = df.set_index("ts").ms.resample("1h").agg(["count", "mean", "max"]).round(0)
    hourly.index = hourly.index.strftime("%m-%d %H시")
    print(hourly.rename(columns={"count": "주기수", "mean": "평균ms", "max": "최대ms"}).to_string())

    print("\n== ③ 공간별 상태 (주기 수) ==")
    st = pd.DataFrame(list(df.status)).apply(lambda c: c.value_counts()).fillna(0).astype(int)
    print(st.to_string())

    print("\n== ④ 경보 발생 내역 (시작/해제/반복) ==")
    ev = [dict(ts=r.ts, **e) for r in df.itertuples() for e in (r.events or [])]
    if ev:
        e = pd.DataFrame(ev)
        e["시각"] = e.ts.dt.strftime("%m-%d %H:%M")
        cols = [c for c in ["시각", "space", "event", "type", "sensor", "detail"] if c in e]
        print(e[cols].to_string(index=False))
        print("\n종류별:", e.groupby(["type", "event"]).size().to_dict())
    else:
        print("없음")


if __name__ == "__main__":
    main()
