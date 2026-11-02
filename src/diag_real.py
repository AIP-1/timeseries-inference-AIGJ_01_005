#!/usr/bin/env python3
"""
과제 2 보강 — 실데이터(real.csv, 9월~) 진단서용 숫자·그래프

  ~/collector/venv/bin/python diag_real.py            # 기본: real.csv, schedule_real.csv
  결과: results_diag/ (summary.txt, *.csv, *.png)

1) 결측률·기록 주기·긴 결측   2) 제거된 이상값 수(과제 3과 같은 규칙)
3) 평일 시간대별 패턴(그래프)  4) 일정 변화 뒤 CO2 반응 시간, 일정과 어긋난 날
5) 기준선 3종 × 5·15·30·60분 (과제 3 모델과 같은 시험 시점)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import forecast as F

ALIAS = {"실습실": "lab", "사무실": "office", "회의실": "meeting"}
HOL = "2026-09-24,2026-09-25,2026-09-26,2026-10-05,2026-10-09"
SPLIT = "2026-09-01,2026-10-07,2026-10-14,2026-10-22"


def workday(ts):
    return ts.weekday() < 5 and ts.strftime("%Y-%m-%d") not in F.HOLIDAYS


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="real.csv")
    ap.add_argument("--schedule", default="schedule_real.csv")
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--holidays", default=HOL)
    ap.add_argument("--out", default="results_diag")
    a = ap.parse_args()
    t = a.split.split(",")
    F.SPLIT = {"train": (t[0], t[1]), "val": (t[1], t[2]), "test": (t[2], t[3])}
    F.HOLIDAYS = set(a.holidays.split(","))
    out = Path(a.out); out.mkdir(exist_ok=True)
    lines = []

    def log(*x):
        s = " ".join(str(i) for i in x); print(s, flush=True); lines.append(s)

    d, sch, flt = F.load(a.csv, a.schedule, None)
    d = d.sort_values("ts")
    start, end = d.ts.min().floor("1min"), d.ts.max().floor("1min")
    total_min = int((end - start).total_seconds() // 60) + 1
    log(f"# 과제 2 보강 — 실데이터 진단  ({start:%m-%d %H:%M} ~ {end:%m-%d %H:%M}, {total_min/1440:.1f}일, {len(d):,}행)")
    log(f"분할: 학습 {F.SPLIT['train']} / 검증 {F.SPLIT['val']} / 시험 {F.SPLIT['test']} | 공휴일 {sorted(F.HOLIDAYS)}\n")

    # ---------------------------------------------------------- 1) 결측과 기록 주기
    log("## 1. 결측률과 기록 주기 (1분 칸 기준)")
    rows = []
    for (sp, m), g in d.groupby(["space", "metric"]):
        ts = g.ts.dt.floor("1min").drop_duplicates()
        gap = ts.diff().dt.total_seconds().div(60).dropna()
        period = float(gap.median()) if len(gap) else np.nan
        # 기록 주기보다 2.5배 넘게 비면 '결측'으로 본다 (주기 때문에 생긴 빈칸은 결측 아님)
        thr = max(2.5 * period, 3)
        miss = gap[gap > thr]
        miss_min = float((miss - period).sum())
        expect = total_min / period
        rows.append(dict(공간=sp, 항목=m, 행수=len(g), 기록주기_분=round(period, 1),
                         기대_대비_수신률=round(len(ts) / expect * 100, 1),
                         결측률=round(miss_min / total_min * 100, 2),
                         짧은결측_10분이하=int((miss <= 10).sum()),
                         중간결측_10분_1시간=int(((miss > 10) & (miss <= 60)).sum()),
                         긴결측_1시간초과=int((miss > 60).sum()),
                         가장긴결측_분=int(miss.max()) if len(miss) else 0))
    miss_t = pd.DataFrame(rows)
    miss_t.to_csv(out / "1_missing.csv", index=False)
    log(miss_t.to_string(index=False))
    # 긴 결측 목록
    long_rows = []
    for (sp, m), g in d.groupby(["space", "metric"]):
        ts = g.ts.dt.floor("1min").drop_duplicates().reset_index(drop=True)
        gap = ts.diff().dt.total_seconds().div(60)
        for i in np.where(gap > 60)[0]:
            long_rows.append(dict(공간=sp, 항목=m, 시작=ts[i - 1], 끝=ts[i], 길이_분=int(gap[i])))
    long_t = pd.DataFrame(long_rows)
    if len(long_t):
        long_t = long_t.sort_values("시작")
        long_t.to_csv(out / "1_long_gaps.csv", index=False)
        # 세 공간이 같은 시각에 함께 빈 구간 = 센서가 아니라 수집·서버 쪽 원인
        log(f"\n긴 결측(1시간 초과) {len(long_t)}건 → 1_long_gaps.csv")
        log(long_t.assign(시작=long_t.시작.dt.strftime("%m-%d %H:%M"), 끝=long_t.끝.dt.strftime("%m-%d %H:%M")).head(20).to_string(index=False))

    # ---------------------------------------------------------- 2) 이상값
    log("\n## 2. 제거된 이상값 (과제 3 정제와 같은 규칙: 범위 밖, 한 칸만 튐)")
    rows = []
    for (sp, m), g in d.groupby(["space", "metric"]):
        v = g.sort_values("ts").value
        lo, hi = F.RANGE[m]
        rng = (v < lo) | (v > hi)
        spike = (v.diff().abs() > F.SPIKE[m]) & (v.diff(-1).abs() > F.SPIKE[m]) & ~rng
        ex = v[rng]
        rows.append(dict(공간=sp, 항목=m, 범위밖=int(rng.sum()), 튐=int(spike.sum()),
                         제거율=round((rng.sum() + spike.sum()) / len(v) * 100, 3),
                         범위밖_예=", ".join(str(x) for x in ex.unique()[:4])))
    out_t = pd.DataFrame(rows)
    out_t.to_csv(out / "2_outliers.csv", index=False)
    log(out_t.to_string(index=False))

    # ---------------------------------------------------------- 5분 격자 (과제 3과 같은 정제)
    grid, idx, spaces = F.clean_grid(d, flt)
    wd = np.array([workday(x) for x in idx])

    # ---------------------------------------------------------- 3) 평일 시간대별 패턴
    log("\n## 3. 평일(공휴일 제외) 시간대별 평균 — 1시간 단위 요약")
    pat = {}
    for sp in spaces:
        g = grid[sp][wd]
        pat[sp] = g.groupby(g.index.strftime("%H:%M")).mean()
    hourly = pd.DataFrame({f"{ALIAS.get(sp, sp)}_co2": pat[sp].co2.groupby(pat[sp].index.str[:2]).mean().round(0) for sp in spaces})
    for sp in spaces:
        hourly[f"{ALIAS.get(sp, sp)}_temp"] = pat[sp].temperature.groupby(pat[sp].index.str[:2]).mean().round(2)
    hourly.to_csv(out / "3_weekday_hourly.csv")
    log(hourly.loc["07":"20"].to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        x = np.arange(288) / 12
        for sp in spaces:
            ax[0].plot(x[:len(pat[sp])], pat[sp].co2.values, label=ALIAS.get(sp, sp))
            ax[1].plot(x[:len(pat[sp])], pat[sp].temperature.values, label=ALIAS.get(sp, sp))
        for axx in ax:
            for s, e in F.CLASS:
                axx.axvspan(int(s[:2]) + int(s[3:]) / 60, int(e[:2]) + int(e[3:]) / 60, color="#e8a33d", alpha=0.12)
            axx.grid(alpha=0.3); axx.legend(loc="upper left")
        ax[0].set_ylabel("CO2 (ppm)"); ax[1].set_ylabel("Temperature (C)"); ax[1].set_xlabel("hour of day")
        ax[0].set_title("Weekday mean (holidays excluded), shaded = lab class time 09:00-11:30, 13:00-18:00")
        ax[1].set_xticks(range(0, 25, 2))
        fig.tight_layout(); fig.savefig(out / "3_weekday_pattern.png", dpi=130); plt.close(fig)
        log("그래프: 3_weekday_pattern.png")
    except Exception as e:
        log(f"그래프 생략: {e}")

    # ---------------------------------------------------------- 4) 일정 변화 뒤 CO2 반응
    log("\n## 4. 일정 변화 뒤 CO2 반응 (일정이 있는 공간)")
    ev_rows, day_rows = [], []
    for sp in sorted(sch.space.unique()):
        if sp not in grid:
            continue
        c = grid[sp].co2
        # 반응 시간은 1분 원본(범위 밖 제거, 3분 중앙값)으로 재야 5분보다 잘게 보인다
        raw = d[(d.space == sp) & (d.metric == "co2")].set_index("ts").value
        raw = raw[(raw >= F.RANGE["co2"][0]) & (raw <= F.RANGE["co2"][1])].resample("1min").mean().interpolate(limit=10, limit_area="inside")
        c1 = raw.rolling(3, center=True, min_periods=1).median()
        for _, r in sch[sch.space == sp].iterrows():
            # 시작: 시작 시각 값보다 30ppm 오를 때까지 몇 분 / 끝: 끝 시각 값보다 30ppm 내릴 때까지 몇 분
            for kind, t0, sign in (("시작", r.start, +1), ("끝", r.end, -1)):
                if t0 not in c1.index or pd.isna(c1[t0]):
                    continue
                w = c1[t0: t0 + pd.Timedelta("90min")]
                hit = w[(w - c1[t0]) * sign >= 30]
                lag = (hit.index[0] - t0).total_seconds() / 60 if len(hit) else np.nan
                ev_rows.append(dict(공간=sp, 종류=kind, 시각=t0, 반응_분=lag,
                                    변화_60분=round(float(c1.get(t0 + pd.Timedelta("60min"), np.nan) - c1[t0]), 0)))
        # 날짜별: 수업 세션 동안 오른 폭 vs 일정 없는 낮(09~18시) 오른 폭
        days = pd.Series(c.index.date).unique()
        for day in days:
            s = sch[(sch.space == sp) & (sch.start.dt.date == day)]
            dd = c[c.index.date == day]
            if dd.notna().sum() < 100:
                continue
            ts0 = pd.Timestamp(day, tz=F.TZ)
            if len(s):
                for _, r in s.iterrows():
                    seg = c[r.start: r.end]
                    if seg.notna().sum() < 6 or pd.isna(c.get(r.start)):
                        continue
                    rise = float(seg.max() - c[r.start])
                    day_rows.append(dict(공간=sp, 날짜=str(day), 요일="월화수목금토일"[ts0.weekday()], 일정="수업",
                                         구간=f"{r.start:%H:%M}-{r.end:%H:%M}", 최대상승_ppm=round(rise),
                                         판정="수업인데 거의 안 오름" if rise < 50 else ""))
            else:
                seg = dd[ts0 + pd.Timedelta("9h"): ts0 + pd.Timedelta("18h")]
                if seg.notna().sum() < 6:
                    continue
                base = float(dd[ts0 + pd.Timedelta("6h"): ts0 + pd.Timedelta("9h")].min())
                rise = float(seg.max() - base)
                day_rows.append(dict(공간=sp, 날짜=str(day), 요일="월화수목금토일"[ts0.weekday()], 일정="없음",
                                     구간="09:00-18:00", 최대상승_ppm=round(rise),
                                     판정="일정 없는데 오름" if rise >= 150 else ""))
    ev = pd.DataFrame(ev_rows)
    days_t = pd.DataFrame(day_rows)
    if len(ev):
        ev.to_csv(out / "4_schedule_events.csv", index=False)
        ev["시각대"] = ev.시각.dt.strftime("%H:%M")
        summ = ev.groupby(["공간", "종류", "시각대"]).agg(
            횟수=("반응_분", "size"), 반응함=("반응_분", lambda x: int(x.notna().sum())),
            반응_중앙값_분=("반응_분", "median"), 반응_25퍼=("반응_분", lambda x: x.quantile(.25)),
            반응_75퍼=("반응_분", lambda x: x.quantile(.75)), 변화60분_중앙값=("변화_60분", "median")).round(1)
        summ.to_csv(out / "4_schedule_response.csv")
        log("반응 = 시작 때보다 30ppm 오르기(끝은 내리기)까지 걸린 분, 90분 안에 없으면 '반응 안 함'")
        log(summ.to_string())
    if len(days_t):
        days_t.to_csv(out / "4_days.csv", index=False)
        odd = days_t[days_t.판정 != ""]
        odd.to_csv(out / "4_mismatch_days.csv", index=False)
        n_cls = (days_t.일정 == "수업").sum()
        log(f"\n일정과 어긋난 구간: 수업 {n_cls}회 중 '거의 안 오름' {(odd.일정 == '수업').sum()}회, "
            f"일정 없는 날 {(days_t.일정 == '없음').sum()}일 중 '오름' {(odd.일정 == '없음').sum()}일 → 4_mismatch_days.csv")
        log(odd.to_string(index=False))

    # ---------------------------------------------------------- 5) 기준선 (과제 3 모델과 같은 시점)
    log("\n## 5. 기준선 3종 (시험 기간, 과제 3 모델과 같은 창·같은 정제)")
    data = F.Data(grid, idx, spaces, sch)
    log("정규화 값(학습 기간만): " + ", ".join(f"{k} 평균 {data.mu[k]:.1f} / 표준편차 {data.sd[k]:.2f}" for k in F.INPUTS))
    te = data.windows("test", 24)
    F.DSD = data.dsd
    base = F.baselines(data, te)
    groups = {"전체": np.ones(len(te["ts"]), bool)}
    for sp in spaces:
        groups[sp] = te["space"] == sp
    if "실습실" in spaces:
        groups["실습실 수업시간"] = (te["space"] == "실습실") & F.in_class(te["ts"]) & np.array([workday(x) for x in te["ts"]])
    res = F.table(base, te, groups)
    res.to_csv(out / "5_baselines.csv", index=False)
    for g in groups:
        sub = res[res.구간 == g]
        if len(sub):
            log(f"\n[{g}] 평가점 {sub.평가점.iloc[0]:,}개")
            log(sub.pivot(index="모델", columns="지평_분", values="CO2_MAE").add_prefix("CO2 ").join(
                sub.pivot(index="모델", columns="지평_분", values="온도_MAE").add_prefix("온도 ")).to_string())

    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n끝. 결과: {out}/  (summary.txt 를 보내 주세요)")


if __name__ == "__main__":
    main()
