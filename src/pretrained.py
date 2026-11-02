#!/usr/bin/env python3
"""
과제 6 — 사전학습 모델(Chronos-Bolt) 비교. 우리 데이터로 추가 학습하지 않는다.

  ~/collector/venv/bin/pip install chronos-forecasting
  ~/collector/venv/bin/python pretrained.py                  # 기본: chronos-bolt-small
  ~/collector/venv/bin/python pretrained.py --model amazon/chronos-bolt-tiny

비교 (과제 3과 같은 시험 주, 같은 시점)
  기준선(현재값 유지) · 직접 학습한 GRU(과제 3) · Chronos
  - 정확도: 5·15·30·60분 뒤 CO2 MAE
  - 예측 구간: Chronos 10~90% 구간 안에 실제가 들어온 비율 (이상적이면 80%)
  - 비용: 1회 추론 시간(공간 1개 / 4개 한꺼번에), 메모리
  - 과거 길이: 2시간 · 12시간 · 1일 · 2일
"""
import argparse
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

import forecast as F

TZ = "Asia/Seoul"
QL = [0.1, 0.5, 0.9]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # 리눅스: KB → MB (최대 사용량)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--every", type=int, default=3, help="시험 시점을 몇 칸마다 고를지 (3 = 15분마다)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(4)

    # ---- 과제 3과 같은 데이터·시점
    d, sch, flt = F.load("synthetic.csv", "schedule.csv", "faults.csv")
    grid, idx, spaces = F.clean_grid(d, flt)
    data = F.Data(grid, idx, spaces, sch)
    F.DSD = data.dsd
    te = data.windows("test", 24)
    base = F.baselines(data, te)
    keep = ~np.isnan(base["어제 같은 시각"]).any((1, 2))
    keep &= (np.arange(len(keep)) % a.every == 0)
    sel = lambda w: {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v) for k, v in w.items()}
    te = sel(te)
    base = {k: v[keep] for k, v in base.items()}
    truth = te["Y0"][:, None, :] + te["Y"] * F.DSD[None, None, :]
    print(f"시험 시점 {len(te['ts']):,}개 (공간 {spaces}, {a.every*5}분마다)")

    # ---- 직접 학습한 GRU (과제 3)
    gru = F.build("gru", te["X"].shape[2], te["F"].shape[2])
    gru.load_state_dict(torch.load("results/gru_main_L24.pt", map_location="cpu")); gru.eval()
    pred_gru = F.predict(gru, te)

    # ---- Chronos
    from chronos import BaseChronosPipeline
    m0 = rss_mb()
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(a.model, device_map="cpu", torch_dtype=torch.float32)
    load_s = time.time() - t0
    m1 = rss_mb()
    co2 = {sp: grid[sp]["co2"] for sp in spaces}

    def context(sp, ts, L):
        s = co2[sp]
        t = data.idx.get_loc(ts)
        c = s.iloc[max(0, t - L + 1): t + 1].interpolate(limit_direction="both")
        return torch.tensor(c.values, dtype=torch.float32)

    rows, cover_rows, preds_c = [], [], {}
    for L, name in ((24, "2시간"), (144, "12시간"), (288, "1일"), (576, "2일")):
        qs = []
        t1 = time.time()
        for i in range(0, len(te["ts"]), 64):
            ctx = [context(sp, ts, L) for sp, ts in zip(te["space"][i:i + 64], te["ts"][i:i + 64])]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=F.H, quantile_levels=QL)
            qs.append(q.numpy())
        q = np.concatenate(qs)                          # [n, 12, 3]
        el = time.time() - t1
        preds_c[name] = q
        for mins, k in F.HORIZ.items():
            y = truth[:, k - 1, 0]
            rows.append(dict(방법=f"Chronos (과거 {name})", 지평_분=mins, CO2_MAE=round(float(np.mean(np.abs(q[:, k - 1, 1] - y))), 1)))
            cover_rows.append(dict(과거=name, 지평_분=mins, 구간적중률=round(float(np.mean((y >= q[:, k - 1, 0]) & (y <= q[:, k - 1, 2]))), 3)))
        print(f"  과거 {name}: {el:.0f}초")
    for name, p in (("현재값 유지", base["현재값 유지"]), ("최근 1시간 평균", base["최근 1시간 평균"]), ("어제 같은 시각", base["어제 같은 시각"]), ("GRU (과제 3)", pred_gru)):
        for mins, k in F.HORIZ.items():
            rows.append(dict(방법=name, 지평_분=mins, CO2_MAE=round(float(np.mean(np.abs(p[:, k - 1, 0] - truth[:, k - 1, 0]))), 1)))
    res = pd.DataFrame(rows).pivot(index="방법", columns="지평_분", values="CO2_MAE")
    cov = pd.DataFrame(cover_rows).pivot(index="과거", columns="지평_분", values="구간적중률")

    # 공간별 60분
    sp_rows = []
    for sp in spaces:
        m = te["space"] == sp
        y = truth[m, 11, 0]
        sp_rows.append(dict(공간=sp, 현재값유지=round(float(np.mean(np.abs(base["현재값 유지"][m, 11, 0] - y))), 1),
                            GRU=round(float(np.mean(np.abs(pred_gru[m, 11, 0] - y))), 1),
                            Chronos_1일=round(float(np.mean(np.abs(preds_c["1일"][m, 11, 1] - y))), 1)))
    spdf = pd.DataFrame(sp_rows)

    # ---- 비용: 1회 추론 시간 (워밍업 5회 후 50회)
    def bench(batch, fn):
        for _ in range(5):
            fn(batch)
        ts = []
        for _ in range(50):
            t = time.perf_counter(); fn(batch); ts.append((time.perf_counter() - t) * 1000)
        return np.mean(ts), np.percentile(ts, 95)
    sample_c = [context(spaces[i % 4], te["ts"][100], 288) for i in range(4)]
    chron = lambda b: pipe.predict_quantiles(sample_c[:b], prediction_length=F.H, quantile_levels=QL)
    xg, fg = torch.from_numpy(te["X"][:4]), torch.from_numpy(te["F"][:4])
    def g(b):
        with torch.no_grad():
            gru(xg[:b], fg[:b])
    cost = []
    for b in (1, 4):
        cm, cp = bench(b, chron); gm, gp = bench(b, g)
        cost.append(dict(배치=b, Chronos_평균ms=round(cm, 1), Chronos_느린5퍼ms=round(cp, 1), GRU_평균ms=round(gm, 2), GRU_느린5퍼ms=round(gp, 2)))
    cost = pd.DataFrame(cost)
    n_params = sum(p.numel() for p in pipe.model.parameters()) if hasattr(pipe, "model") else None
    g_params = sum(p.numel() for p in gru.parameters())

    res.to_csv(out / "pretrained_mae.csv"); cov.to_csv(out / "pretrained_coverage.csv"); cost.to_csv(out / "pretrained_cost.csv", index=False)
    spdf.to_csv(out / "pretrained_by_space.csv", index=False)
    print("\n== CO2 MAE (ppm, 같은 시점) ==\n" + res.to_string())
    print("\n== 공간별 60분 CO2 MAE ==\n" + spdf.to_string(index=False))
    print("\n== Chronos 10~90% 구간 안에 실제가 들어온 비율 (이상 0.80) ==\n" + cov.to_string())
    print("\n== 1회 추론 시간 (CPU, 워밍업 5회 후 50회) ==\n" + cost.to_string(index=False))
    print(f"\n메모리: Chronos 불러오기 전 최대 {m0:.0f}MB → 후 {m1:.0f}MB (+{m1-m0:.0f}MB), 불러오기 {load_s:.1f}초")
    print(f"파라미터 수: Chronos {n_params:,} / GRU {g_params:,}" if n_params else f"파라미터 수: GRU {g_params:,}")


if __name__ == "__main__":
    main()
