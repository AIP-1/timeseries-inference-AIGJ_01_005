#!/usr/bin/env python3
"""
학교 InfluxDB(iot-bucket, 읽기 전용)에서 우리 센서 3개의 옛날 데이터를 받아
과제 3 학습용 CSV 로 저장한다. (젯슨에 랜선이 꽂혀 있어야 함)

  set -a; source ~/collector/.env; set +a
  ~/collector/venv/bin/python influx_export.py --start 2026-09-01

만드는 파일
  real.csv           ts,space,metric,value  (1분 평균)
  schedule_real.csv  실습실 수업 시간 (평일 09:00~11:30, 13:00~18:00, 공휴일 제외)
"""
import argparse
import csv
import io
import os
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone

URL = "http://10.116.64.13:8086/api/v2/query?org=33e75366a5081f13"
BUCKET = "iot-bucket"
DEVICES = {"실습실": "24e124128c067999", "사무실": "24e124725d081175", "회의실": "24e124725d089152"}
METRICS = ["co2", "temperature", "humidity"]
KST = timezone(timedelta(hours=9))
CLASS = [("09:00", "11:30"), ("13:00", "18:00")]


def token():
    t = os.environ.get("INFLUX_TOKEN")
    if not t:
        for line in open(os.path.expanduser("~/collector/.env"), encoding="utf-8"):
            if line.startswith("INFLUX_TOKEN="):
                t = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not t:
        raise SystemExit("INFLUX_TOKEN 이 없어요. 먼저: set -a; source ~/collector/.env; set +a")
    return t


def query(flux, tok):
    req = urllib.request.Request(URL, data=flux.encode(), method="POST", headers={
        "Authorization": f"Token {tok}", "Accept": "application/csv", "Content-type": "application/vnd.flux"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read().decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--stop", default=None, help="기본: 지금")
    ap.add_argument("--holidays", default="2026-09-24,2026-09-25,2026-09-26,2026-10-05,2026-10-09")
    ap.add_argument("--out", default="real.csv")
    a = ap.parse_args()
    tok = token()
    start = datetime.fromisoformat(a.start).replace(tzinfo=KST)
    stop = datetime.fromisoformat(a.stop).replace(tzinfo=KST) if a.stop else datetime.now(KST)

    n_total = 0
    with open(a.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["ts", "space", "metric", "value"])
        for space, dev in DEVICES.items():
            for m in METRICS:
                t0, n, locs = time.time(), 0, set()
                cur = start
                while cur < stop:                       # 2주씩 나눠서 받기 (한 번에 너무 크지 않게)
                    nxt = min(cur + timedelta(days=14), stop)
                    flux = (f'from(bucket:"{BUCKET}") |> range(start:{cur.isoformat()}, stop:{nxt.isoformat()}) '
                            f'|> filter(fn:(r)=> r._measurement=="{m}" and r.device_id=="{dev}" and r._field=="value") '
                            f'|> group(columns:["location"]) '
                            f'|> aggregateWindow(every:1m, fn:mean, createEmpty:false) '
                            f'|> keep(columns:["_time","_value","location"])')
                    text = query(flux, tok)
                    rd = csv.DictReader(io.StringIO(text))
                    for row in rd:
                        if not row.get("_time") or row.get("_value") in (None, ""):
                            continue
                        ts = datetime.fromisoformat(row["_time"].replace("Z", "+00:00")) - timedelta(minutes=1)   # 창 끝 → 창 시작
                        w.writerow([ts.astimezone(KST).isoformat(), space, m, row["_value"]])
                        locs.add(row.get("location", ""))
                        n += 1
                    cur = nxt
                n_total += n
                print(f"  {space} {m}: {n:,}줄 ({time.time()-t0:.0f}초) 위치표시 {sorted(locs)}", flush=True)

    # 실습실 시간표
    hol = set(a.holidays.split(","))
    with open("schedule_real.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["space", "start", "end", "kind"])
        d = start.date()
        while d <= stop.date():
            if d.weekday() < 5 and d.isoformat() not in hol:
                for s, e in CLASS:
                    w.writerow(["실습실", f"{d.isoformat()}T{s}:00+09:00", f"{d.isoformat()}T{e}:00+09:00", "수업"])
            d += timedelta(days=1)
    print(f"\n완료: {a.out} {n_total:,}줄, schedule_real.csv (실습실 수업, 공휴일 {sorted(hol)} 제외)")
    print(f"기간: {start:%Y-%m-%d} ~ {stop:%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
