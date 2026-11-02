#!/usr/bin/env python3
"""
학교 InfluxDB(iot-bucket, 읽기 전용)에 어떤 데이터가 얼마나 있는지 둘러본다. 랜선 필요.

  ~/collector/venv/bin/python influx_explore.py            # 측정 이름·태그·필드·기간
  ~/collector/venv/bin/python influx_explore.py --days 60  # 최근 60일 기준으로 셈
"""
import argparse
import io
import os
import urllib.request
from pathlib import Path

import pandas as pd

URL, ORG, BUCKET = "http://10.116.64.13:8086", "33e75366a5081f13", "iot-bucket"


def token():
    for p in (Path.home() / "collector" / ".env", Path(".env")):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("INFLUX_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("INFLUX_TOKEN", "")


def flux(q, tok, timeout=120):
    req = urllib.request.Request(f"{URL}/api/v2/query?org={ORG}", data=q.encode(), method="POST",
                                 headers={"Authorization": f"Token {tok}", "Content-Type": "application/vnd.flux",
                                          "Accept": "application/csv"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode()
    parts = [p for p in txt.split("\r\n\r\n") if p.strip()]
    dfs = [pd.read_csv(io.StringIO(p)) for p in parts]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df.drop(columns=[c for c in df.columns if c in ("Unnamed: 0", "result", "table")], errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    a = ap.parse_args()
    tok = token()
    if not tok:
        raise SystemExit("INFLUX_TOKEN 을 ~/collector/.env 에서 못 찾았어요")
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)

    m = flux(f'import "influxdata/influxdb/schema"\nschema.measurements(bucket: "{BUCKET}", start: -{a.days}d)', tok)
    names = list(m["_value"]) if len(m) else []
    print(f"== 측정 이름 (최근 {a.days}일): {names}")
    for name in names:
        tags = flux(f'import "influxdata/influxdb/schema"\nschema.measurementTagKeys(bucket: "{BUCKET}", measurement: "{name}", start: -{a.days}d)', tok)
        fields = flux(f'import "influxdata/influxdb/schema"\nschema.measurementFieldKeys(bucket: "{BUCKET}", measurement: "{name}", start: -{a.days}d)', tok)
        print(f"\n--- {name}\n태그: {[t for t in tags['_value'] if not t.startswith('_')] if len(tags) else []}")
        print(f"필드: {list(fields['_value']) if len(fields) else []}")
        for tag in [t for t in (tags["_value"] if len(tags) else []) if not t.startswith("_")][:6]:
            vals = flux(f'import "influxdata/influxdb/schema"\nschema.measurementTagValues(bucket: "{BUCKET}", measurement: "{name}", tag: "{tag}", start: -{a.days}d)', tok)
            v = list(vals["_value"]) if len(vals) else []
            print(f"  {tag} ({len(v)}개): {v[:15]}{' …' if len(v) > 15 else ''}")
        # 필드별 기간과 건수 (co2·temperature·humidity 만)
        for f in [x for x in (fields["_value"] if len(fields) else []) if str(x).lower() in ("co2", "temperature", "humidity")]:
            q = f'''from(bucket: "{BUCKET}") |> range(start: -{a.days}d)
  |> filter(fn: (r) => r._measurement == "{name}" and r._field == "{f}") |> group()'''
            first = flux(q + " |> first()", tok); last = flux(q + " |> last()", tok); cnt = flux(q + " |> count()", tok)
            print(f"  [{f}] {first['_time'].iloc[0] if len(first) else '-'} ~ {last['_time'].iloc[0] if len(last) else '-'}, "
                  f"{int(cnt['_value'].iloc[0]) if len(cnt) else 0:,}건")
    # 샘플 몇 줄
    if names:
        s = flux(f'from(bucket: "{BUCKET}") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "{names[0]}") |> limit(n: 2)', tok)
        print("\n== 샘플 (최근 1시간, 첫 측정)\n" + s.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
