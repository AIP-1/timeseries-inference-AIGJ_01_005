#!/usr/bin/env python3
"""
과제 5 — 상태 추정
  ① 사용 여부: CO2·온도·습도 변화만 보고 '사용중/비어있음' (일정은 입력에 넣지 않음)
  ② 환기 선제 경보: 지금은 1,000ppm 미만인데 30분 안에 넘을 것인가

  python3 state.py           # synthetic.csv, schedule.csv, occupancy_truth.csv, faults.csv → results/state_*.csv

정답
  사용 여부(학습용)  = 일정표(수업·예약) 기반. 현장에서 실제로 구할 수 있는 정답이라서.
  사용 여부(평가용)  = 실제 인원(occupancy_truth.csv) ≥ 1명. 과제의 '직접 관찰'에 해당.
  환기 경보 양성     = 지금 CO2 < 1,000 이고 다음 30분 안에 1,000 이상. 이미 넘은 시점은 평가에서 뺀다.
분할: 학습 09-21~10-04 / 검증 10-05~10-11(기준 정하기) / 시험 10-12~10-18. 고장 구간은 뺀다.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
T = lambda s: pd.Timestamp(s, tz=TZ)
TRAIN, VAL, TEST = (T("2026-09-21"), T("2026-10-05")), (T("2026-10-05"), T("2026-10-12")), (T("2026-10-12"), T("2026-10-19"))
HOLIDAYS = {"2026-10-09"}
METRICS = ["co2", "temperature", "humidity"]
LIMIT = 1000


def load(a):
    d = pd.read_csv(a.csv)
    d["ts"] = pd.to_datetime(d.ts, utc=True, format="ISO8601").dt.tz_convert(TZ)
    sch = pd.read_csv(a.schedule)
    flt = pd.read_csv(a.faults)
    for x in (sch, flt):
        for c in ("start", "end"):
            x[c] = pd.to_datetime(x[c], utc=True, format="ISO8601").dt.tz_convert(TZ)
    occ = pd.read_csv(a.truth, index_col=0)
    occ.index = pd.to_datetime(occ.index, utc=True, format="ISO8601").tz_convert(TZ)
    return d, sch, flt, occ


def frame(d, sch, flt, occ, space):
    """공간 하나의 5분 표: 센서, 변화량 특징, 정답들."""
    g = d[d.space == space].pivot_table(index="ts", columns="metric", values="value").resample("5min").mean()[METRICS]
    g = g.interpolate(limit=2, limit_area="inside")
    f = pd.DataFrame(index=g.index)
    f["co2"], f["temp"], f["hum"] = g.co2, g.temperature, g.humidity
    for k, name in ((1, "5"), (3, "15"), (6, "30"), (12, "60")):
        f[f"dco2_{name}"] = g.co2 - g.co2.shift(k)
        f[f"dtemp_{name}"] = g.temperature - g.temperature.shift(k)
    f["dhum_15"] = g.humidity - g.humidity.shift(3)
    f["co2_floor"] = g.co2.rolling(144, min_periods=36).min()       # 최근 12시간 바닥값 → 밤사이 빈 방 수준
    f["co2_above_floor"] = g.co2 - f.co2_floor
    f["dco2_slope_change"] = f.dco2_15 - f.dco2_15.shift(3)          # 상승이 꺾였나
    h = (g.index.hour + g.index.minute / 60) / 24 * 2 * np.pi
    f["hsin"], f["hcos"] = np.sin(h), np.cos(h)                      # 환기 경보에만 사용
    f["work"] = [float(t.weekday() < 5 and t.strftime("%Y-%m-%d") not in HOLIDAYS) for t in g.index]
    busy = np.zeros(len(f))
    soon = np.zeros(len(f))
    for _, r in sch[sch.space == space].iterrows():
        busy[(f.index >= r.start) & (f.index < r.end)] = 1
        soon[(f.index >= r.start - pd.Timedelta("30min")) & (f.index < r.end)] = 1
    f["sched_busy"], f["sched_soon"] = busy, soon                    # 환기 경보에만 사용
    # 정답
    f["label_sched"] = busy.astype(int)
    f["label_obs"] = (occ[space].reindex(f.index) >= 0.5).astype(int)
    fut = pd.concat([g.co2.shift(-k) for k in range(1, 7)], axis=1).max(axis=1)
    f["vent_pos"] = ((g.co2 < LIMIT) & (fut >= LIMIT)).astype(int)
    f["vent_valid"] = (g.co2 < LIMIT) & fut.notna()                 # 이미 넘은 시점은 제외
    # 고장 구간(+회복 3시간) 제외
    bad = np.zeros(len(f), bool)
    for _, r in flt[flt.space == space].iterrows():
        bad |= (f.index >= r.start) & (f.index < r.end + pd.Timedelta("3h"))
    f["bad"] = bad
    f["space"] = space
    return f


OCC_X = ["co2", "temp", "hum", "dco2_5", "dco2_15", "dco2_30", "dco2_60", "dtemp_15", "dtemp_30", "dtemp_60",
         "dhum_15", "co2_above_floor", "dco2_slope_change"]                       # 일정·시각 없음
VENT_X = ["co2", "dco2_5", "dco2_15", "dco2_30", "dco2_slope_change", "co2_above_floor", "temp", "dtemp_30",
          "hsin", "hcos", "work", "sched_busy", "sched_soon"]


def split(df, rng):
    return df[(df.index >= rng[0]) & (df.index < rng[1]) & ~df.bad].dropna(subset=OCC_X + VENT_X)


def onehot(df, spaces):
    return np.stack([(df.space == s).astype(float) for s in spaces], 1)


def report(y, p):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
    cm = confusion_matrix(y, p, labels=[0, 1])
    return dict(정확도=round(accuracy_score(y, p), 3), F1=round(f1_score(y, p), 3), 정밀도=round(precision_score(y, p, zero_division=0), 3),
                재현율=round(recall_score(y, p), 3), macroF1=round(f1_score(y, p, average="macro"), 3),
                빈방을사용중으로=round(cm[0, 1] / max(cm[0].sum(), 1), 3), cm=cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="synthetic.csv")
    ap.add_argument("--schedule", default="schedule.csv")
    ap.add_argument("--faults", default="faults.csv")
    ap.add_argument("--truth", default="occupancy_truth.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier

    d, sch, flt, occ = load(a)
    spaces = sorted(d.space.unique())
    allf = pd.concat([frame(d, sch, flt, occ, s) for s in spaces])
    tr, va, te = split(allf, TRAIN), split(allf, VAL), split(allf, TEST)

    # ---------------- ① 사용 여부
    print("== ① 사용 여부 ==")
    # 일정 기반 정답이 실제(관찰)와 얼마나 다른가
    rows = []
    for s in spaces:
        x = allf[(allf.space == s) & (allf.label_obs.notna())]
        x = x[x.work == 1]
        both = pd.crosstab(x.label_sched, x.label_obs)
        def c(i, j):
            return int(both.loc[i, j]) if i in both.index and j in both.columns else 0
        rows.append(dict(공간=s, 일치율=round((x.label_sched == x.label_obs).mean(), 3),
                         일정있는데_빈방_분=c(1, 0) * 5, 일정없는데_사용_분=c(0, 1) * 5))
    gap = pd.DataFrame(rows)
    gap.to_csv(out / "state_schedule_vs_observed.csv", index=False)
    print("일정 기반 정답 vs 실제(평일):\n" + gap.to_string(index=False))

    Xtr = np.hstack([tr[OCC_X].values, onehot(tr, spaces)])
    Xte = np.hstack([te[OCC_X].values, onehot(te, spaces)])
    occ_res = []
    m_sched = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr, tr.label_sched)
    m_obs = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr, tr.label_obs)
    preds = {
        "단순 규칙 (15분 CO2 +30ppm)": (te.dco2_15 >= 30).astype(int).values,
        "분류기 (일정 정답으로 학습)": m_sched.predict(Xte),
        "분류기 (관찰 정답으로 학습)": m_obs.predict(Xte),
    }
    # 일정을 입력에 넣으면? (정답 누출 시연)
    Xtr_leak = np.hstack([Xtr, tr[["sched_busy"]].values]); Xte_leak = np.hstack([Xte, te[["sched_busy"]].values])
    m_leak = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr_leak, tr.label_sched)
    preds["(시연) 일정을 입력에도 넣음"] = m_leak.predict(Xte_leak)
    cms = {}
    for name, p in preds.items():
        r_obs = report(te.label_obs.values, p)
        r_sch = report(te.label_sched.values, p)
        cms[name] = r_obs.pop("cm"); r_sch.pop("cm")
        occ_res.append(dict(방법=name, **{f"관찰기준_{k}": v for k, v in r_obs.items()}, 일정기준_정확도=r_sch["정확도"]))
    occ_df = pd.DataFrame(occ_res)
    occ_df.to_csv(out / "state_occupancy.csv", index=False)
    print("\n시험 주 성적 (관찰 정답 기준):\n" + occ_df[["방법", "관찰기준_정확도", "관찰기준_F1", "관찰기준_빈방을사용중으로", "일정기준_정확도"]].to_string(index=False))
    best = "분류기 (일정 정답으로 학습)"
    print(f"\n혼동 행렬 [{best}] (행=실제 비어있음/사용중, 열=예측):\n{cms[best]}")
    print(f"혼동 행렬 [단순 규칙]:\n{cms['단순 규칙 (15분 CO2 +30ppm)']}")
    # 가장 많이 틀리는 상황
    te2 = te.copy(); te2["pred"] = preds[best]; te2["err"] = te2.pred != te2.label_obs
    occ_prev = te2.groupby("space").label_obs.transform(lambda s: s.rolling(6, min_periods=1).max().shift(1))   # 30분 안에 사용중이었나
    te2["상황"] = np.select(
        [(te2.label_obs == 0) & (occ_prev == 1), (te2.label_obs == 1) & (te2.groupby("space").label_obs.transform(lambda s: s.rolling(3, min_periods=1).min().shift(1)) == 0),
         (te2.label_obs == 0), (te2.label_obs == 1)],
        ["막 나간 직후 30분(빈방)", "막 들어온 직후 15분(사용중)", "그 밖의 빈방", "그 밖의 사용중"], "기타")
    sit = te2.groupby(["상황"]).agg(칸수=("err", "size"), 오류율=("err", "mean")).round(3).sort_values("오류율", ascending=False)
    sp_err = te2.groupby("space").err.mean().round(3)
    sit.to_csv(out / "state_occupancy_errors.csv")
    print("\n상황별 오류율:\n" + sit.to_string())
    print("\n공간별 오류율:\n" + sp_err.to_string())

    # ---------------- ② 환기 선제 경보
    print("\n== ② 환기 선제 경보 (30분 안에 1,000ppm) ==")
    vtr, vva, vte = (x[x.vent_valid] for x in (tr, va, te))
    print(f"양성 비율: 학습 {vtr.vent_pos.mean()*100:.1f}%, 시험 {vte.vent_pos.mean()*100:.1f}% ({vte.vent_pos.sum()}칸)")
    Xv = lambda x: np.hstack([x[VENT_X].values, onehot(x, spaces)])
    w = np.where(vtr.vent_pos == 1, (1 - vtr.vent_pos.mean()) / vtr.vent_pos.mean(), 1.0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xv(vtr), vtr.vent_pos, sample_weight=w)
    # 기준: 검증 주에서 재현율 90%를 넘기는 가장 높은 확률 기준 (시험 주는 안 봄)
    pv = clf.predict_proba(Xv(vva))[:, 1]
    thr = 0.5
    for t in np.linspace(0.95, 0.05, 91):
        if ((pv >= t) & (vva.vent_pos == 1)).sum() / max(vva.vent_pos.sum(), 1) >= 0.90:
            thr = t; break
    pt = clf.predict_proba(Xv(vte))[:, 1]
    vp = {
        "전부 '경보 없음'": np.zeros(len(vte), int),
        "단순 규칙 (CO2 ≥ 850)": (vte.co2 >= 850).astype(int).values,
        f"분류기 (확률 ≥ {thr:.2f})": (pt >= thr).astype(int),
    }
    vres, vcms = [], {}
    for name, p in vp.items():
        r = report(vte.vent_pos.values, p)
        vcms[name] = r.pop("cm")
        vres.append(dict(방법=name, **{k: v for k, v in r.items() if k in ("정확도", "재현율", "정밀도", "macroF1")}))
    vdf = pd.DataFrame(vres)
    vdf.to_csv(out / "state_vent_alarm.csv", index=False)
    print(vdf.to_string(index=False))
    print(f"\n혼동 행렬 [분류기]:\n{vcms[list(vp)[2]]}\n혼동 행렬 [단순 규칙]:\n{vcms[list(vp)[1]]}")
    vte2 = vte.copy(); vte2["p"] = vp[list(vp)[2]]
    fp = vte2[(vte2.p == 1) & (vte2.vent_pos == 0)]
    fn = vte2[(vte2.p == 0) & (vte2.vent_pos == 1)]
    print(f"\n헛경보 {len(fp)}칸: 공간별 {fp.space.value_counts().to_dict()}, CO2 중앙값 {fp.co2.median():.0f}ppm")
    print(f"놓침 {len(fn)}칸: 공간별 {fn.space.value_counts().to_dict()}, 15분 상승 중앙값 {fn.dco2_15.median():.0f}ppm")


if __name__ == "__main__":
    main()
