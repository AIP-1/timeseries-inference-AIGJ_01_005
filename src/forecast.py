#!/usr/bin/env python3
"""
과제 3 — 60분 예측 (5분 간격 12칸, CO2·온도)

  venv/bin/python forecast.py                       # 기본 실험: GRU·LSTM·TCN x 일정 포함/제외 (과거 2시간)
  venv/bin/python forecast.py --exp hist            # 심화: 과거 1·2·4시간 x 모델 3종
  venv/bin/python forecast.py --exp holdout --holdout 회의실   # 심화: 처음 보는 공간
  venv/bin/python forecast.py --dry                 # torch 없이 데이터·평가 경로만 점검

입력 (매 5분 칸마다)
  과거 L칸: CO2·온도·습도(학습 구간 평균·표준편차로 정규화), 하루 중 시각(sin·cos), 근무일 여부,
            지금 일정 있음(0/1), 공간(원-핫)
  미래 12칸: 하루 중 시각, 근무일 여부, 일정 있음(0/1)  ← 시간표·예약은 미리 알 수 있는 정보
출력: 60분을 한 번에 — 12칸 x (CO2, 온도)의 '지금 값에서의 변화량'
분할: 학습 09-21~10-04 / 검증 10-05~10-11 / 시험 10-12~10-18. 정규화·모델 선택은 학습·검증만 씀.
고장 구간(faults.csv)은 학습과 예측 평가에서 뺀다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
STEP = "5min"
H = 12                                   # 60분 / 5분
HORIZ = {5: 1, 15: 3, 30: 6, 60: 12}     # 분 → 칸
TARGETS = ["co2", "temperature"]
INPUTS = ["co2", "temperature", "humidity"]
RANGE = {"co2": (350, 5000), "temperature": (5, 40), "humidity": (5, 100)}
SPIKE = {"co2": 300, "temperature": 3.0, "humidity": 15}
GAP = {"co2": 2, "temperature": 6, "humidity": 6}      # 보간할 최대 칸(10분, 30분)
SPLIT = {"train": ("2026-09-21", "2026-10-05"), "val": ("2026-10-05", "2026-10-12"), "test": ("2026-10-12", "2026-10-19")}
HOLIDAYS = {"2026-10-09"}
CLASS = [("09:00", "11:30"), ("13:00", "18:00")]


# ------------------------------------------------------------------ 데이터 준비 (과제 2 정제 규칙)
def load(csv, schedule_csv, faults_csv):
    d = pd.read_csv(csv)
    d["ts"] = pd.to_datetime(d.ts, utc=True, format="ISO8601").dt.tz_convert(TZ)
    d = d[d.metric.isin(INPUTS)]
    sch = pd.read_csv(schedule_csv)
    for c in ("start", "end"):
        sch[c] = pd.to_datetime(sch[c], utc=True, format="ISO8601").dt.tz_convert(TZ)
    flt = pd.read_csv(faults_csv) if faults_csv and Path(faults_csv).exists() else pd.DataFrame(columns=["space", "metric", "start", "end", "kind"])
    for c in ("start", "end"):
        flt[c] = pd.to_datetime(flt[c], utc=True, format="ISO8601").dt.tz_convert(TZ)
    return d, sch, flt


def clean_grid(d, flt):
    """공간별 5분 격자 표. 범위·튐 제거 → 고장 구간 제거 → 5분 평균 → 짧은 결측 보간."""
    spaces = sorted(d.space.unique())
    idx = pd.date_range(d.ts.min().floor(STEP), d.ts.max().floor(STEP), freq=STEP)
    grid = {}
    for sp in spaces:
        cols = {}
        for m in INPUTS:
            g = d[(d.space == sp) & (d.metric == m)].sort_values("ts")
            v = g.value
            lo, hi = RANGE[m]
            bad = (v < lo) | (v > hi) | ((v.diff().abs() > SPIKE[m]) & (v.diff(-1).abs() > SPIKE[m]))
            g = g[~bad]
            for _, f in flt[(flt.space == sp) & (flt.metric.isin([m, "*"]))].iterrows():
                g = g[(g.ts < f.start) | (g.ts >= f.end)]
            s = g.set_index("ts").value.resample(STEP).mean().reindex(idx)
            cols[m] = s.interpolate(limit=GAP[m], limit_area="inside")
        grid[sp] = pd.DataFrame(cols)
    return grid, idx, spaces


def schedule_flag(idx, sch, space):
    f = np.zeros(len(idx), np.float32)
    for _, r in sch[sch.space == space].iterrows():
        f[(idx >= r.start) & (idx < r.end)] = 1
    return f


def calendar(idx):
    h = (idx.hour + idx.minute / 60) / 24 * 2 * np.pi
    work = np.array([(t.weekday() < 5 and t.strftime("%Y-%m-%d") not in HOLIDAYS) for t in idx], np.float32)
    return np.stack([np.sin(h), np.cos(h), work], 1).astype(np.float32)


def in_class(idx):
    hm = idx.strftime("%H:%M")
    m = np.zeros(len(idx), bool)
    for a, b in CLASS:
        m |= (hm >= a) & (hm < b)
    return m & (idx.weekday < 5)


# ------------------------------------------------------------------ 창(window) 만들기
class Data:
    def __init__(self, grid, idx, spaces, sch, use_schedule=True, use_space=True):
        self.grid, self.idx, self.spaces = grid, idx, spaces
        self.cal = calendar(idx)
        self.sch = {sp: schedule_flag(idx, sch, sp) for sp in spaces}
        self.use_schedule, self.use_space = use_schedule, use_space
        tr = (idx >= pd.Timestamp(SPLIT["train"][0], tz=TZ)) & (idx < pd.Timestamp(SPLIT["train"][1], tz=TZ))
        allv = pd.concat([grid[sp][tr] for sp in spaces])
        self.mu = allv.mean()                     # 정규화: 학습 구간만
        self.sd = allv.std()
        dl = []
        for sp in spaces:
            g = grid[sp][TARGETS]
            for k in range(1, H + 1):
                dl.append((g.shift(-k) - g)[tr])
        self.dsd = pd.concat(dl).std().values.astype(np.float32)    # 변화량 크기(학습 구간)

    def arrays(self, sp):
        g = self.grid[sp]
        x = ((g[INPUTS] - self.mu[INPUTS]) / self.sd[INPUTS]).values.astype(np.float32)
        parts = [x, self.cal]
        if self.use_schedule:
            parts.append(self.sch[sp][:, None])
        if self.use_space:
            oh = np.zeros((len(self.idx), len(self.spaces)), np.float32)
            oh[:, self.spaces.index(sp)] = 1
            parts.append(oh)
        hist = np.concatenate(parts, 1)
        fut = [self.cal] + ([self.sch[sp][:, None]] if self.use_schedule else [])
        return hist, np.concatenate(fut, 1), g[TARGETS].values.astype(np.float32)

    def windows(self, split, L, spaces=None):
        a, b = (pd.Timestamp(x, tz=TZ) for x in SPLIT[split])
        X, F, Y, Y0, meta = [], [], [], [], []
        for sp in spaces or self.spaces:
            hist, fut, raw = self.arrays(sp)
            ok = ~np.isnan(hist).any(1)
            okr = ~np.isnan(raw).any(1)
            for t in range(L - 1, len(self.idx) - H):
                ts = self.idx[t]
                # 입력 시작부터 60분 뒤 정답까지 모두 이 구간 안에 있어야 함 → 경계에서 정답이 새지 않음
                if self.idx[t - L + 1] < a or self.idx[t + H] >= b:
                    continue
                if not ok[t - L + 1: t + 1].all() or not okr[t: t + H + 1].all():
                    continue
                X.append(hist[t - L + 1: t + 1]); F.append(fut[t + 1: t + H + 1])
                Y.append((raw[t + 1: t + H + 1] - raw[t]) / self.dsd); Y0.append(raw[t])
                meta.append((sp, ts))
        if not X:
            return None
        return dict(X=np.stack(X), F=np.stack(F), Y=np.stack(Y), Y0=np.stack(Y0),
                    space=np.array([m[0] for m in meta]), ts=pd.DatetimeIndex([m[1] for m in meta]))


# ------------------------------------------------------------------ 기준선 (과제 2와 같은 정의)
def baselines(data, w):
    """w 의 각 시점에서 세 기준선의 12칸 예측(원 단위)."""
    out = {"현재값 유지": np.repeat(w["Y0"][:, None, :], H, 1)}
    mean1h, day = [], []
    for sp, ts in zip(w["space"], w["ts"]):
        g = data.grid[sp][TARGETS]
        t = data.idx.get_loc(ts)
        mean1h.append(np.repeat(np.nanmean(g.values[t - 11: t + 1], 0)[None], H, 0))
        prev = t + np.arange(1, H + 1) - 288
        day.append(g.values[prev] if prev.min() >= 0 else np.full((H, 2), np.nan))
    out["최근 1시간 평균"] = np.stack(mean1h)
    out["어제 같은 시각"] = np.stack(day)
    return out


def score(pred, w, mask=None):
    """지평별 MAE (CO2, 온도)."""
    truth = w["Y0"][:, None, :] + w["Y"] * DSD[None, None, :]
    m = np.ones(len(truth), bool) if mask is None else mask
    res = {}
    for mins, k in HORIZ.items():
        e = np.abs(pred[m, k - 1, :] - truth[m, k - 1, :])
        res[mins] = np.nanmean(e, 0)
    return res


# ------------------------------------------------------------------ 모델
def build(kind, n_in, n_fut, hidden=64):
    import torch
    import torch.nn as nn

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c = [], n_in
            for dil in (1, 2, 4, 8):
                layers += [nn.ConstantPad1d((2 * dil, 0), 0), nn.Conv1d(c, hidden, 3, dilation=dil), nn.ReLU()]
                c = hidden
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x.transpose(1, 2))[:, :, -1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "gru":
                self.enc = nn.GRU(n_in, hidden, batch_first=True)
            elif kind == "lstm":
                self.enc = nn.LSTM(n_in, hidden, batch_first=True)
            else:
                self.enc = TCN()
            self.head = nn.Sequential(nn.Linear(hidden + H * n_fut, 128), nn.ReLU(), nn.Linear(128, H * 2))

        def forward(self, x, f):
            if kind == "tcn":
                h = self.enc(x)
            else:
                o, _ = self.enc(x)
                h = o[:, -1]
            return self.head(torch.cat([h, f.flatten(1)], 1)).view(-1, H, 2)

    return Net()


def train(kind, tr, va, epochs=40, patience=6, seed=0, log=print):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(kind, tr["X"].shape[2], tr["F"].shape[2])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    T = lambda a: torch.from_numpy(a)
    Xt, Ft, Yt = T(tr["X"]), T(tr["F"]), T(tr["Y"])
    Xv, Fv, Yv = T(va["X"]), T(va["F"]), T(va["Y"])
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))               # 창 순서만 섞음(창 안의 시간 순서는 그대로)
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]
            loss = torch.nn.functional.smooth_l1_loss(model(Xt[j], Ft[j]), Yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.smooth_l1_loss(model(Xv, Fv), Yv).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 5 == 0 or wait == 0:
            log(f"    {kind} epoch {ep:2d} val {vl:.4f}{' *' if wait == 0 else ''}")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, best


def predict(model, w):
    import torch
    with torch.no_grad():
        d = model(torch.from_numpy(w["X"]), torch.from_numpy(w["F"])).numpy()
    return w["Y0"][:, None, :] + d * DSD[None, None, :]


# ------------------------------------------------------------------ 실험
def table(results, w, groups):
    rows = []
    for name, pred in results.items():
        for gname, mask in groups.items():
            if mask.sum() == 0:
                continue
            s = score(pred, w, mask)
            for mins in HORIZ:
                rows.append(dict(모델=name, 구간=gname, 지평_분=mins, CO2_MAE=round(float(s[mins][0]), 1),
                                 온도_MAE=round(float(s[mins][1]), 3), 평가점=int(mask.sum())))
    return pd.DataFrame(rows)


def main():
    global DSD
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="synthetic.csv")
    ap.add_argument("--schedule", default="schedule.csv")
    ap.add_argument("--faults", default="faults.csv")
    ap.add_argument("--exp", default="main", choices=["main", "hist", "holdout"])
    ap.add_argument("--hist", type=int, default=24, help="과거 칸 수(5분 단위). 24 = 2시간")
    ap.add_argument("--holdout", default="회의실")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dry", action="store_true", help="torch 없이 데이터·평가만 점검")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default=None, help="실데이터용: 학습시작,검증시작,시험시작,시험끝 (예: 2026-09-01,2026-10-07,2026-10-14,2026-10-22)")
    ap.add_argument("--holidays", default=None, help="쉼표로 구분한 공휴일 날짜")
    a = ap.parse_args()
    global SPLIT, HOLIDAYS
    if a.split:
        t = a.split.split(",")
        SPLIT = {"train": (t[0], t[1]), "val": (t[1], t[2]), "test": (t[2], t[3])}
    if a.holidays:
        HOLIDAYS = set(a.holidays.split(","))
    out = Path(a.out); out.mkdir(exist_ok=True)
    logf = open(out / f"log_{a.exp}.txt", "w", encoding="utf-8")
    def log(*x):
        s = " ".join(str(i) for i in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"분할: 학습 {SPLIT['train']} / 검증 {SPLIT['val']} / 시험 {SPLIT['test']} | 공휴일 {sorted(HOLIDAYS)}")

    t0 = time.time()
    d, sch, flt = load(a.csv, a.schedule, a.faults)
    grid, idx, spaces = clean_grid(d, flt)
    log(f"데이터 {idx[0]:%m-%d} ~ {idx[-1]:%m-%d}, 공간 {spaces}, 고장 구간 {len(flt)}개 제외")

    if not a.dry:
        import torch
        torch.set_num_threads(4)

    def run(models, L, use_schedule, train_spaces=None, test_spaces=None, use_space=True, tag=""):
        global DSD
        data = Data(grid, idx, spaces, sch, use_schedule, use_space)
        DSD = data.dsd
        tr = data.windows("train", L, train_spaces); va = data.windows("val", L, train_spaces)
        te = data.windows("test", L, test_spaces)
        log(f"[{tag}] 과거 {L*5}분, 일정 {'포함' if use_schedule else '제외'}: 학습 {len(tr['X']):,} / 검증 {len(va['X']):,} / 시험 {len(te['X']):,} 창, 입력 {tr['X'].shape[2]}개")
        preds = {}
        for kind in models:
            if a.dry:
                preds[kind.upper()] = predict_dry(te)
                continue
            t1 = time.time()
            m, vl = train(kind, tr, va, a.epochs, log=log)
            preds[f"{kind.upper()}{tag}"] = predict(m, te)
            log(f"  {kind}: 검증 손실 {vl:.4f}, {time.time()-t1:.0f}초")
            torch.save(m.state_dict(), out / f"{kind}_{a.exp}{tag}_L{L}.pt")
        return data, te, preds

    if a.exp == "main":
        data, te, p_s = run(["gru", "lstm", "tcn"], a.hist, True, tag="")
        _, te2, p_n = run(["gru", "lstm", "tcn"], a.hist, False, tag="-일정제외")
        assert (te["ts"] == te2["ts"]).all() and (te["space"] == te2["space"]).all(), "같은 시점에서 비교해야 함"
        DSD = data.dsd
        base = baselines(data, te)
        allp = {**base, **p_s, **p_n}
        # 세 기준선이 모두 있는 같은 시점만 평가
        common = ~np.isnan(base["어제 같은 시각"]).any((1, 2))
        groups = {"전체": common}
        for sp in spaces:
            groups[sp] = common & (te["space"] == sp)
        for cls_sp in ("강의실A", "실습실"):
            if cls_sp in spaces:
                groups[f"{cls_sp} 수업 시간"] = common & (te["space"] == cls_sp) & in_class(te["ts"])
        res = table(allp, te, groups)
        res.to_csv(out / "main_results.csv", index=False)
        log("\n== 시험 기간 MAE (전체) ==")
        log(res[res.구간 == "전체"].pivot(index="모델", columns="지평_분", values="CO2_MAE").to_string())
        log(res[res.구간 == "전체"].pivot(index="모델", columns="지평_분", values="온도_MAE").to_string())
        log("\n== 공간별 60분 CO2 MAE ==")
        log(res[res.지평_분 == 60].pivot(index="모델", columns="구간", values="CO2_MAE").to_string())

    elif a.exp == "hist":
        runs = {L: run(["gru", "lstm", "tcn"], L, True, tag=f"-L{L}") for L in (12, 24, 48)}
        # 과거 길이가 달라도 같은 시점에서 비교: 세 실험에 모두 있는 (공간, 시각)만
        keys = [set(zip(te["space"], te["ts"])) for _, te, _ in runs.values()]
        common = set.intersection(*keys)
        rows = []
        for L, (data, te, p) in runs.items():
            DSD = data.dsd
            keep = np.array([k in common for k in zip(te["space"], te["ts"])])
            r = table(p, te, {"전체": keep})
            r["과거_분"] = L * 5
            rows.append(r)
        res = pd.concat(rows)
        res.to_csv(out / "hist_results.csv", index=False)
        log(res[res.지평_분 == 60].pivot(index="모델", columns="과거_분", values="CO2_MAE").to_string())

    else:  # holdout
        others = [s for s in spaces if s != a.holdout]
        data, te_seen, p_seen = run(["gru"], a.hist, True, tag="-전체공간", test_spaces=[a.holdout], use_space=False)
        _, te_new, p_new = run(["gru"], a.hist, True, tag=f"-{a.holdout}제외", train_spaces=others, test_spaces=[a.holdout], use_space=False)
        DSD = data.dsd
        base = baselines(data, te_new)
        m = np.ones(len(te_new["ts"]), bool)
        res = table({**base, **p_seen, **p_new}, te_new, {a.holdout: m})
        res.to_csv(out / f"holdout_{a.holdout}.csv", index=False)
        log(res.pivot(index="모델", columns="지평_분", values="CO2_MAE").to_string())

    log(f"\n끝 ({time.time()-t0:.0f}초). 결과: {out}/")


def predict_dry(w):
    return np.repeat(w["Y0"][:, None, :], H, 1)       # 변화량 0 = 현재값 유지 (경로 점검용)


DSD = None
if __name__ == "__main__":
    main()
