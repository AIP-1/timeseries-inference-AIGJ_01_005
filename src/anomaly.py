#!/usr/bin/env python3
"""
과제 4 — 이상 탐지 (정상만 보고 배우는 오토인코더 + 규칙)

  python3 anomaly.py                  # synthetic.csv, faults.csv → results/anomaly_*.csv

모델  : 오토인코더(입력을 압축했다가 다시 그려 보는 신경망). 학습 2주(정상)만 보고 배운다.
        1시간(5분 x 12칸)의 CO2·온도·습도 + 시각·근무일 + 공간 → 같은 1시간을 복원.
        복원이 안 될수록 '평소와 다르다'. 가장 복원이 안 된 항목 = 원인 센서.
규칙  : 튐(직전 10분 중앙값과 크게 다름), 값 고정(정상 기간 최장 기록의 1.5배 넘게 같은 값),
        범위 밖, 새벽 기준선(사람 없는 02~05시 CO2가 평소보다 100ppm 이상 높음 → 영점 틀어짐, 심화).
임계값: 고장이 없는 검증 주(10-05~10-11)에서 정한다. 시험 주(고장 15건)는 평가에만 쓴다.
평가  : 고장 종류별 탐지 건수, 탐지 지연, 하루당 헛경보(전체 공간 합).
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
W = 12                       # 1시간 창
SPIKE = {"co2": 400, "temperature": 3.0, "humidity": 12}
RANGE = {"co2": (350, 5000), "temperature": (5, 40), "humidity": (5, 100)}
GRACE = pd.Timedelta("30min")    # 고장이 끝난 뒤 30분 안의 경보까지 그 고장을 잡은 것으로 봄
RECOVER = {"냉방정지": pd.Timedelta("3h"), "환기정지": pd.Timedelta("2h")}   # 설비가 돌아와도 방이 원래대로 돌아오는 데 걸리는 시간


def load(csv, faults_csv):
    d = pd.read_csv(csv)
    d["ts"] = pd.to_datetime(d.ts, utc=True, format="ISO8601").dt.tz_convert(TZ)
    d = d[d.metric.isin(METRICS)]
    f = pd.read_csv(faults_csv)
    for c in ("start", "end"):
        f[c] = pd.to_datetime(f[c], utc=True, format="ISO8601").dt.tz_convert(TZ)
    return d, f


def wide(d, space, rule):
    g = d[d.space == space].pivot_table(index="ts", columns="metric", values="value")
    return g.resample(rule).mean()[METRICS] if rule != "1min" else g.reindex(columns=METRICS)


# ------------------------------------------------------------------ 경보 → 사건(에피소드)
def episodes(flag, idx, min_len=1, merge="30min"):
    """참인 칸이 min_len 칸 이상 이어지면 경보. merge 안에 다시 울리면 같은 사건으로 묶음."""
    ep, start, run = [], None, 0
    for t, f in zip(idx, flag):
        if f:
            run += 1
            if run == min_len:
                start = t - pd.Timedelta(minutes=5 * (min_len - 1)) if min_len > 1 else t
        else:
            if start is not None:
                ep.append([start, t]); start = None
            run = 0
    if start is not None:
        ep.append([start, idx[-1]])
    out = []
    for a, b in ep:
        if out and a - out[-1][1] <= pd.Timedelta(merge):
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


# ------------------------------------------------------------------ 규칙
def rules(d, space, normal_runs, night_base):
    """1분 데이터에 규칙 적용. → {규칙명: [(시작, 끝, 원인 항목)]}"""
    g = wide(d, space, "1min")
    res = {"튐": [], "값고정": [], "범위밖": [], "새벽기준선": []}
    for m in METRICS:
        s = g[m].dropna()
        med = s.rolling("10min", closed="left").median()
        spike = (s - med).abs() > SPIKE[m]
        for a, b in episodes(spike.values, s.index, merge="10min"):
            res["튐"].append((a, b, m))
        lo, hi = RANGE[m]
        for a, b in episodes(((s < lo) | (s > hi)).values, s.index, merge="10min"):
            res["범위밖"].append((a, b, m))
        # 값 고정: 같은 값이 이어진 시간(분)
        grp = s.ne(s.shift()).cumsum()
        start = s.index.to_series().groupby(grp.values).transform("first")
        runmin = (s.index.to_series() - start).dt.total_seconds() / 60
        lim = normal_runs[(space, m)] * 1.5
        for a, b in episodes((runmin > lim).values, s.index, merge="10min"):
            res["값고정"].append((a - pd.Timedelta(minutes=lim), b, m))
    # 새벽 기준선(심화): 02~05시 CO2 중앙값이 학습 기간 평소보다 100ppm 이상 높으면 그날 05시에 경보
    co2 = g["co2"].dropna()
    night = co2[(co2.index.hour >= 2) & (co2.index.hour < 5)]
    for day, v in night.groupby(night.index.date):
        if v.median() - night_base[space] > 100:
            t = T(f"{day} 05:00")
            res["새벽기준선"].append((t, t + pd.Timedelta("5min"), "co2"))
    return res


def normal_stats(d):
    runs, base = {}, {}
    tr = d[(d.ts >= TRAIN[0]) & (d.ts < TRAIN[1])]
    for sp in tr.space.unique():
        g = wide(tr, sp, "1min")
        for m in METRICS:
            s = g[m].dropna()
            grp = s.ne(s.shift()).cumsum()
            dur = s.index.to_series().groupby(grp.values).agg(lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds() / 60)
            runs[(sp, m)] = max(float(dur.max()), 10.0)
        c = g["co2"].dropna()
        base[sp] = float(c[(c.index.hour >= 2) & (c.index.hour < 5)].median())
    return runs, base


# ------------------------------------------------------------------ 오토인코더
class AE:
    def __init__(self, d, spaces):
        self.spaces = spaces
        self.grids = {sp: wide(d, sp, "5min").interpolate(limit=6, limit_area="inside") for sp in spaces}
        tr = pd.concat([g[(g.index >= TRAIN[0]) & (g.index < TRAIN[1])] for g in self.grids.values()])
        self.mu, self.sd = tr.mean(), tr.std()          # 정규화: 학습 구간만

    def features(self, sp):
        g = self.grids[sp]
        z = ((g - self.mu) / self.sd).values.astype(np.float32)
        idx = g.index
        h = (idx.hour + idx.minute / 60) / 24 * 2 * np.pi
        work = np.array([(t.weekday() < 5 and t.strftime("%Y-%m-%d") not in HOLIDAYS) for t in idx], np.float32)
        dow = np.eye(7, dtype=np.float32)[idx.weekday]
        ctx = np.concatenate([np.stack([np.sin(h), np.cos(h), work], 1), dow], 1)
        oh = np.zeros(len(self.spaces), np.float32); oh[self.spaces.index(sp)] = 1
        X, keep = [], []
        for t in range(W - 1, len(idx)):
            win = z[t - W + 1: t + 1]
            if np.isnan(win).any():
                continue
            X.append(np.concatenate([win.ravel(), ctx[t], oh])); keep.append(t)
        return np.array(X), idx[keep]

    def fit(self, seed=0):
        from sklearn.neural_network import MLPRegressor
        Xs = []
        for sp in self.spaces:
            X, ts = self.features(sp)
            Xs.append(X[(ts >= TRAIN[0]) & (ts < TRAIN[1])])
        X = np.concatenate(Xs)
        self.n = W * len(METRICS)
        self.model = MLPRegressor(hidden_layer_sizes=(64, 12, 64), activation="relu", max_iter=200,
                                  early_stopping=True, random_state=seed, learning_rate_init=1e-3)
        self.model.fit(X, X[:, :self.n])                 # 정상 1시간을 다시 그리도록
        return self

    def score(self, sp):
        X, ts = self.features(sp)
        R = self.model.predict(X)
        err = (R - X[:, :self.n]) ** 2
        per = err.reshape(len(X), W, len(METRICS))
        last = per[:, -3:, :].mean(1)                    # 최근 15분의 항목별 복원 오차
        return pd.DataFrame(last, index=ts, columns=METRICS)


# ------------------------------------------------------------------ 예측 오차 탐지기 (심화)
class ForecastError:
    """2시간 전까지의 1시간 + 지금 시각·요일·일정 → 지금 값을 예측. 실제가 예측에서 벗어난 정도 = 이상 점수.
    오토인코더가 못 보는 '천천히, 그럴듯한 범위 안에서' 벗어나는 설비 고장(냉방·환기 정지)을 겨냥한다."""
    LAG = 24    # 2시간 전까지만 보고 지금을 예측 → 몇 시간에 걸쳐 쌓이는 설비 고장도 보임

    def __init__(self, ae, schedule):
        self.ae = ae
        self.sch = schedule

    def features(self, sp):
        g = self.ae.grids[sp]
        z = ((g - self.ae.mu) / self.ae.sd).values.astype(np.float32)
        idx = g.index
        h = (idx.hour + idx.minute / 60) / 24 * 2 * np.pi
        work = np.array([(t.weekday() < 5 and t.strftime("%Y-%m-%d") not in HOLIDAYS) for t in idx], np.float32)
        busy = np.zeros(len(idx), np.float32)
        for _, r in self.sch[self.sch.space == sp].iterrows():
            busy[(idx >= r.start) & (idx < r.end)] = 1
        oh = np.zeros(len(self.ae.spaces), np.float32); oh[self.ae.spaces.index(sp)] = 1
        X, Y, keep = [], [], []
        for t in range(W + self.LAG - 1, len(idx)):
            hist = z[t - self.LAG - W + 1: t - self.LAG + 1]
            if np.isnan(hist).any() or np.isnan(z[t]).any():
                continue
            ctx = np.concatenate([[np.sin(h[t]), np.cos(h[t]), work[t]], np.eye(7)[idx[t].weekday()], busy[t - self.LAG + 1: t + 1]])
            X.append(np.concatenate([hist.ravel(), ctx, oh])); Y.append(z[t] - z[t - self.LAG]); keep.append(t)
        return np.array(X), np.array(Y), idx[keep]

    def fit(self, seed=0):
        from sklearn.neural_network import MLPRegressor
        Xs, Ys = [], []
        for sp in self.ae.spaces:
            X, Y, ts = self.features(sp)
            m = (ts >= TRAIN[0]) & (ts < TRAIN[1])
            Xs.append(X[m]); Ys.append(Y[m])
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=seed)
        self.model.fit(np.concatenate(Xs), np.concatenate(Ys))
        # 정상(학습) 잔차의 크기로 항목·공간별 눈금을 맞춤
        self.scale = {}
        for sp in self.ae.spaces:
            X, Y, ts = self.features(sp)
            m = (ts >= TRAIN[0]) & (ts < TRAIN[1])
            self.scale[sp] = np.abs(self.model.predict(X[m]) - Y[m]).mean(0) + 1e-6
        return self

    def score(self, sp):
        X, Y, ts = self.features(sp)
        r = (Y - self.model.predict(X)) / self.scale[sp]            # 부호 있는 잔차(정상 평균 1 눈금)
        s = pd.DataFrame(np.abs(r), index=ts, columns=METRICS)
        return s.rolling(3, min_periods=3).mean().dropna() ** 2      # 15분 내내 벗어나야 큼


# ------------------------------------------------------------------ 평가
def merge_alarms(alarms, gap="30min"):
    """같은 공간에서 겹치거나 30분 안에 이어진 경보는 알림 하나로 묶음(탐지기가 달라도). 원인은 먼저 울린 쪽."""
    out = []
    for sp in sorted({a[0] for a in alarms}):
        al = sorted([a for a in alarms if a[0] == sp], key=lambda x: x[1])
        for a in al:
            if out and out[-1][0] == sp and a[1] <= out[-1][2] + pd.Timedelta(gap):
                out[-1] = (sp, out[-1][1], max(out[-1][2], a[2]), out[-1][3])
            else:
                out.append(a)
    return out


def evaluate(alarms, faults, n_days):
    """alarms: [(공간, 시작, 끝, 원인)] → 종류별 성적, 헛경보/일"""
    alarms = merge_alarms(alarms)
    rows, used = [], set()
    for _, f in faults.iterrows():
        hit = [(i, a) for i, a in enumerate(alarms) if a[0] == f.space and a[1] <= f.end + GRACE and a[2] >= f.start]
        # 고장 직후 회복 중에 울린 경보는 헛경보로 세지 않음(탐지로도 세지 않음)
        rec = [i for i, a in enumerate(alarms) if a[0] == f.space and f.end <= a[1] <= f.end + RECOVER.get(f.kind, GRACE)]
        used.update(rec)
        if hit:
            first = min(hit, key=lambda x: x[1][1])
            used.update(i for i, _ in hit)
            delay = max(0, (first[1][1] - f.start).total_seconds() / 60)
            rows.append(dict(종류=f.kind, 공간=f.space, 시작=f"{f.start:%m-%d %H:%M}", 탐지=1, 지연_분=round(delay),
                             원인맞음=int(first[1][3] == f.metric)))
        else:
            rows.append(dict(종류=f.kind, 공간=f.space, 시작=f"{f.start:%m-%d %H:%M}", 탐지=0, 지연_분=None, 원인맞음=0))
    fa = [a for i, a in enumerate(alarms) if i not in used]
    return pd.DataFrame(rows), len(fa) / n_days, fa


def model_alarms(scores, thr, idx_filter):
    out = []
    for sp, s in scores.items():
        s = s[idx_filter(s.index)]
        ratio = s / thr[sp]                                   # 항목마다 따로 기준
        for a, b in episodes((ratio > 1).any(axis=1).values, s.index, min_len=6):   # 30분 이상 이어져야 경보 (설비 고장은 몇 시간씩 이어짐, 짧은 튐은 규칙이 잡음)
            cause = ratio[(ratio.index >= a) & (ratio.index <= b)].max().idxmax()
            out.append((sp, a, b, cause))
    return out


def rule_alarms(rule_res, idx_filter, which=("튐", "값고정", "범위밖", "새벽기준선")):
    out = []
    for sp, res in rule_res.items():
        for k in which:
            for a, b, m in res[k]:
                if idx_filter(pd.DatetimeIndex([a]))[0]:
                    out.append((sp, a, b, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="synthetic.csv")
    ap.add_argument("--faults", default="faults.csv")
    ap.add_argument("--schedule", default="schedule.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    d, faults = load(a.csv, a.faults)
    spaces = sorted(d.space.unique())
    in_val = lambda ix: (ix >= VAL[0]) & (ix < VAL[1])
    in_test = lambda ix: (ix >= TEST[0]) & (ix < TEST[1])

    runs, base = normal_stats(d)
    print("정상 기간 최장 같은 값(분):", {f"{k[0]}/{k[1]}": round(v) for k, v in runs.items()})
    rr = {sp: rules(d, sp, runs, base) for sp in spaces}

    sch = pd.read_csv(a.schedule)
    for c in ("start", "end"):
        sch[c] = pd.to_datetime(sch[c], utc=True, format="ISO8601").dt.tz_convert(TZ)
    ae = AE(d, spaces).fit()
    fe = ForecastError(ae, sch).fit()
    detectors = {"오토인코더": {sp: ae.score(sp) for sp in spaces}, "예측오차": {sp: fe.score(sp) for sp in spaces}}

    # 임계값: 검증 주(정상) 점수의 분위수. 공간마다 따로. 시험 주는 보지 않고 고른다.
    sweeps, chosen = [], {}
    for dname, scores in detectors.items():
        rows = []
        for q in (0.99, 0.995, 0.998, 0.999, 0.9995, 1.0):
            thr = {sp: scores[sp][in_val(scores[sp].index)].quantile(q) * (1.0 if q < 1 else 1.1) for sp in spaces}
            fa_val = len(model_alarms(scores, thr, in_val)) / 7
            res, fa_test, _ = evaluate(model_alarms(scores, thr, in_test), faults, 7)
            rows.append(dict(탐지기=dname, 기준=f"검증 상위 {100-q*100:.2f}%" if q < 1 else "검증 최댓값 x1.1",
                             검증_헛경보_일=round(fa_val, 2), 시험_탐지=f"{res.탐지.sum()}/{len(res)}",
                             시험_헛경보_일=round(fa_test, 2), thr=thr))
        sw = pd.DataFrame(rows)
        pick = sw.iloc[-1]   # 검증 주 정상 점수 최댓값 x1.1: 검증 기간 헛경보 0 + 계절 변화 대비 10% 여유 (시험 주는 안 봄)
        chosen[dname] = pick.thr
        sweeps.append(sw)
        print(f"\n== 임계값별: {dname} ==\n" + sw.drop(columns="thr").to_string(index=False))
        print(f"선택: {pick.기준} (검증 주 헛경보 {pick.검증_헛경보_일}/일)")
    sw = pd.concat(sweeps)

    # 결합(규칙 + 두 탐지기) 기준으로 임계값을 함께 바꿔 보기
    r_all_val, r_all_test = rule_alarms(rr, in_val), rule_alarms(rr, in_test)
    comb = []
    qs = [0.99, 0.995, 0.998, 0.999, 0.9995, 1.0]
    for q in qs:
        th = {dn: {sp: detectors[dn][sp][in_val(detectors[dn][sp].index)].quantile(q) * (1.0 if q < 1 else 1.1) for sp in spaces} for dn in detectors}
        al_v = r_all_val + sum((model_alarms(detectors[dn], th[dn], in_val) for dn in detectors), [])
        al_t = r_all_test + sum((model_alarms(detectors[dn], th[dn], in_test) for dn in detectors), [])
        fa_v = len(merge_alarms(al_v)) / 7
        res, fa_t, _ = evaluate(al_t, faults, 7)
        comb.append(dict(기준=f"검증 상위 {100-q*100:.2f}%" if q < 1 else "검증 최댓값 x1.1", 검증_헛경보_일=round(fa_v, 2),
                         시험_탐지율=f"{res.탐지.mean()*100:.0f}% ({res.탐지.sum()}/{len(res)})", 시험_헛경보_일=round(fa_t, 2)))
    comb = pd.DataFrame(comb)
    comb.to_csv(out / "anomaly_threshold_sweep_combined.csv", index=False)
    print("\n== 임계값별: 결합(규칙+두 탐지기) ==\n" + comb.to_string(index=False))

    ae_al = model_alarms(detectors["오토인코더"], chosen["오토인코더"], in_test)
    fe_al = model_alarms(detectors["예측오차"], chosen["예측오차"], in_test)
    r_al = rule_alarms(rr, in_test)
    r_al_basic = rule_alarms(rr, in_test, ("튐", "값고정", "범위밖"))
    m_al = ae_al + fe_al
    table = []
    for name, al in (("오토인코더", ae_al), ("예측오차(심화)", fe_al), ("규칙", r_al_basic), ("규칙+새벽기준선(심화)", r_al),
                     ("결합: 오토인코더+규칙", ae_al + r_al_basic), ("결합: 전부", m_al + r_al)):
        res, fa, _ = evaluate(al, faults, 7)
        res.to_csv(out / f"anomaly_events_{name}.csv", index=False)
        by = res.groupby("종류").탐지.agg(lambda x: f"{x.sum()}/{len(x)}")
        row = dict(방법=name, **by.to_dict(), 전체=f"{res.탐지.sum()}/{len(res)} ({res.탐지.mean()*100:.0f}%)",
                   평균지연_분=round(res.지연_분.dropna().mean()) if res.탐지.any() else None, 헛경보_일=round(fa, 2),
                   원인맞음=f"{res.원인맞음.sum()}/{res.탐지.sum()}")
        table.append(row)
    tab = pd.DataFrame(table)
    tab.to_csv(out / "anomaly_scorecard.csv", index=False)
    sw.drop(columns="thr").to_csv(out / "anomaly_threshold_sweep.csv", index=False)
    print("\n== 탐지 성적표 (시험 주, 고장 15건) ==\n" + tab.to_string(index=False))
    res, fa, fal = evaluate(m_al + r_al, faults, 7)
    print("\n== 결합: 고장별 ==\n" + res.to_string(index=False))
    print(f"\n결합 헛경보 {len(fal)}건:", [(x[0], f"{x[1]:%m-%d %H:%M}", x[3]) for x in fal])


if __name__ == "__main__":
    main()
