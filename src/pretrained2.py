#!/usr/bin/env python3
"""
과제 6 심화 — ① 공변량(온습도·일정)을 받는 사전학습 모델  ② GRU + 사전학습 모델 섞기

  ~/collector/venv/bin/pip install -U "chronos-forecasting>=2.0" --no-deps    # Chronos-2 가 들어 있는 판
  ~/collector/venv/bin/python pretrained2.py --check     # 2개 시점만 돌려 API·모양 확인 (1~2분)
  ~/collector/venv/bin/python pretrained2.py             # 본 실험 (합성 데이터, 과제 6과 같은 시점)

비교 (과제 6과 같은 2,028개 시점, 과거 1일)
  현재값 유지 · GRU(과제 3) · Chronos-Bolt-small
  · GRU+Bolt 평균(반반) · GRU+Bolt 가중 평균(가중치는 검증 주에서 고름)
  · Chronos-2 CO2만 · Chronos-2 + 공변량
      과거 공변량: 온도, 습도, 일정 있음, 하루 중 시각(sin·cos), 근무일
      미래 공변량: 일정 있음, 하루 중 시각(sin·cos), 근무일  ← 미리 알 수 있는 것만
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import forecast as F

QL = [0.1, 0.5, 0.9]
L = 288                     # 과거 1일 (과제 6에서 Bolt 가 가장 좋았던 길이 중 하나)


def to_q(out):
    """파이프라인마다 다른 출력 모양을 [n, 12, 3] 으로 맞춘다."""
    arr = []
    for o in out:
        a = o.detach().cpu().numpy() if hasattr(o, "detach") else np.asarray(o)
        a = np.squeeze(a)
        if a.shape == (len(QL), F.H):
            a = a.T
        arr.append(a)
    return np.stack(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bolt", default="amazon/chronos-bolt-small")
    ap.add_argument("--c2", default="amazon/chronos-2")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(4)

    d, sch, flt = F.load("synthetic.csv", "schedule.csv", "faults.csv")
    grid, idx, spaces = F.clean_grid(d, flt)
    data = F.Data(grid, idx, spaces, sch)
    F.DSD = data.dsd

    def prep(split, every):
        w = data.windows(split, 24)
        base = F.baselines(data, w)
        keep = ~np.isnan(base["어제 같은 시각"]).any((1, 2))
        keep &= (np.arange(len(keep)) % every == 0)
        w = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v) for k, v in w.items()}
        base = {k: v[keep] for k, v in base.items()}
        truth = w["Y0"][:, None, :] + w["Y"] * F.DSD[None, None, :]
        return w, base, truth

    te, base, truth = prep("test", a.every)
    va, _, truth_va = prep("val", a.every * 2)
    if a.check:
        te = {k: (v[:2] if hasattr(v, "__len__") and len(v) == len(truth) else v) for k, v in te.items()}
        truth, base = truth[:2], {k: v[:2] for k, v in base.items()}
        va = {k: (v[:2] if hasattr(v, "__len__") and len(v) == len(truth_va) else v) for k, v in va.items()}
        truth_va = truth_va[:2]
    print(f"시험 시점 {len(truth):,}개, 검증 시점 {len(truth_va):,}개 (공간 {spaces})", flush=True)

    gru = F.build("gru", te["X"].shape[2], te["F"].shape[2])
    gru.load_state_dict(torch.load("results/gru_main_L24.pt", map_location="cpu")); gru.eval()
    p_gru, p_gru_va = F.predict(gru, te), F.predict(gru, va)

    co2 = {sp: grid[sp]["co2"] for sp in spaces}
    temp = {sp: grid[sp]["temperature"] for sp in spaces}
    hum = {sp: grid[sp]["humidity"] for sp in spaces}

    def seg(s, t0, t1):
        return s.iloc[max(0, t0): t1].interpolate(limit_direction="both").values.astype(np.float32)

    def ctx_target(sp, ts):
        t = data.idx.get_loc(ts)
        return torch.tensor(seg(co2[sp], t - L + 1, t + 1))

    def ctx_cov(sp, ts, with_cov):
        t = data.idx.get_loc(ts)
        lo = max(0, t - L + 1)
        item = {"target": seg(co2[sp], lo, t + 1)}
        if with_cov:
            sc, cal = data.sch[sp], data.cal
            item["past_covariates"] = {
                "temperature": seg(temp[sp], lo, t + 1), "humidity": seg(hum[sp], lo, t + 1),
                "schedule": sc[lo: t + 1].astype(np.float32), "hsin": cal[lo: t + 1, 0], "hcos": cal[lo: t + 1, 1],
                "work": cal[lo: t + 1, 2]}
            item["future_covariates"] = {
                "schedule": sc[t + 1: t + 1 + F.H].astype(np.float32), "hsin": cal[t + 1: t + 1 + F.H, 0],
                "hcos": cal[t + 1: t + 1 + F.H, 1], "work": cal[t + 1: t + 1 + F.H, 2]}
        return item

    from chronos import BaseChronosPipeline

    def run(pipe, w, make, bs=64):
        qs, t0 = [], time.time()
        for i in range(0, len(w["ts"]), bs):
            batch = [make(sp, ts) for sp, ts in zip(w["space"][i:i + bs], w["ts"][i:i + bs])]
            q, _ = pipe.predict_quantiles(batch, prediction_length=F.H, quantile_levels=QL)
            qs.append(to_q(q))
        return np.concatenate(qs), time.time() - t0

    # ---- Chronos-Bolt (과제 6과 같은 모델)
    bolt = BaseChronosPipeline.from_pretrained(a.bolt, device_map="cpu", torch_dtype=torch.float32)
    q_bolt, s_bolt = run(bolt, te, ctx_target)
    q_bolt_va, _ = run(bolt, va, ctx_target)
    print(f"  Bolt: 시험 {s_bolt:.0f}초", flush=True)

    # ---- ② 섞기: 반반, 그리고 검증 주에서 고른 가중치 (CO2 60분 MAE 최소)
    blend = lambda wgt, g, c: wgt * g[:, :, 0] + (1 - wgt) * c[:, :, 1]
    grid_w = np.round(np.arange(0, 1.01, 0.1), 1)
    va_mae = [np.mean(np.abs(blend(wv, p_gru_va, q_bolt_va)[:, 11] - truth_va[:, 11, 0])) for wv in grid_w]
    w_best = float(grid_w[int(np.argmin(va_mae))])
    print(f"  검증 주에서 고른 GRU 가중치 {w_best} (GRU {w_best:.0%} + Bolt {1-w_best:.0%})", flush=True)

    preds = {"현재값 유지": base["현재값 유지"][:, :, 0], "GRU (과제 3)": p_gru[:, :, 0],
             "Chronos-Bolt (과거 1일)": q_bolt[:, :, 1],
             "GRU+Bolt 반반": blend(0.5, p_gru, q_bolt),
             f"GRU+Bolt 가중 (GRU {w_best:.1f})": blend(w_best, p_gru, q_bolt)}
    cover, cost = {}, {"Chronos-Bolt": s_bolt / len(truth) * 1000}

    # ---- ① Chronos-2: CO2만 / 공변량 포함
    try:
        c2 = BaseChronosPipeline.from_pretrained(a.c2, device_map="cpu", torch_dtype=torch.float32)
        for with_cov, name in ((False, "Chronos-2 CO2만"), (True, "Chronos-2 + 공변량")):
            q, s = run(c2, te, lambda sp, ts, wc=with_cov: ctx_cov(sp, ts, wc), bs=32)
            if a.check:
                print(f"  [{name}] 출력 모양 {q.shape}, 첫 시점 60분 뒤 10/50/90% = {q[0, 11]} (실제 {truth[0, 11, 0]:.0f})")
            preds[f"{name} (과거 1일)"] = q[:, :, 1]
            cover[name] = q
            cost[name] = s / len(truth) * 1000
            print(f"  {name}: {s:.0f}초", flush=True)
    except Exception as e:
        import traceback
        print(f"\n  Chronos-2 실패: {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
    cover["Chronos-Bolt"] = q_bolt

    rows = []
    for name, p in preds.items():
        r = dict(방법=name)
        for mins, k in F.HORIZ.items():
            r[f"{mins}분"] = round(float(np.mean(np.abs(p[:, k - 1] - truth[:, k - 1, 0]))), 1)
        rows.append(r)
    res = pd.DataFrame(rows).set_index("방법")
    sp_rows = []
    for sp in spaces:
        m = te["space"] == sp
        sp_rows.append({"공간": sp, **{n: round(float(np.mean(np.abs(p[m, 11] - truth[m, 11, 0]))), 1) for n, p in preds.items()}})
    spdf = pd.DataFrame(sp_rows).set_index("공간").T
    cov_rows = [dict(모델=n, **{f"{mins}분": round(float(np.mean((truth[:, k - 1, 0] >= q[:, k - 1, 0]) & (truth[:, k - 1, 0] <= q[:, k - 1, 2]))), 3)
                                for mins, k in F.HORIZ.items()}) for n, q in cover.items()]
    covdf = pd.DataFrame(cov_rows).set_index("모델")
    costdf = pd.DataFrame([dict(모델=k, 시점당_ms=round(v, 1)) for k, v in cost.items()])

    if not a.check:
        res.to_csv(out / "pretrained2_mae.csv"); spdf.to_csv(out / "pretrained2_by_space.csv")
        covdf.to_csv(out / "pretrained2_coverage.csv"); costdf.to_csv(out / "pretrained2_cost.csv", index=False)
    print("\n== CO2 MAE (ppm, 과제 6과 같은 시점) ==\n" + res.to_string())
    print("\n== 공간별 60분 CO2 MAE ==\n" + spdf.to_string())
    print("\n== 10~90% 구간 적중률 (이상 0.80) ==\n" + covdf.to_string())
    print("\n== 시점 1개당 걸린 시간 (배치로 돌린 전체 시간 ÷ 시점 수, CPU) ==\n" + costdf.to_string(index=False))
    print(f"\n가중치 고르기(검증 주 60분 MAE): " + ", ".join(f"{wv}:{m:.1f}" for wv, m in zip(grid_w, va_mae)))


if __name__ == "__main__":
    main()
