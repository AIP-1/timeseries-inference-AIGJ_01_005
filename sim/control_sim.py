#!/usr/bin/env python3
"""
과제 8 심화 ① — 예측으로 환기 제어 규칙을 설계하고, 지난 1주(시험 주) 실데이터에 적용했다면
1,000ppm 초과 시간이 얼마나 줄었을지 추정한다.

  ~/collector/venv/bin/python control_sim.py            # 실데이터 + 실데이터로 학습한 GRU(results_real)

방법 (5분 칸, 공간마다)
  1) 방의 CO2 물질 수지:  다음 CO2 = 지금 + 5분 × (사람이 내는 양 − 환기율 × (지금 − 바깥))
  2) 환기율(평소)은 사람이 없는 밤(22~06시) CO2가 빠지는 속도로 추정, 바깥 수준은 기간 중 하위 2%
  3) 실제 기록에서 '사람이 내는 양'을 칸마다 거꾸로 계산 (실제 기록 = 제어 없음)
  4) 제어가 켜진 칸은 환기율을 BOOST 배로 올려 CO2를 다시 계산 (사람이 내는 양은 그대로)
  5) 규칙 3가지를 같은 조건에서 비교
       없음      : 실제 기록 그대로
       뒤늦게     : 지금 CO2 ≥ 1,000 이면 켬, 900 아래로 내려가면 끔
       예측       : GRU가 30분 안에 1,000 이상을 예측하거나 지금 ≥ 1,000 이면 켬,
                    예측 최댓값과 지금이 모두 900 아래면 끔
     예측 규칙은 매 칸, 제어가 반영된 '가상의 과거'로 GRU를 다시 돌린다(닫힌 고리).
지표: 1,000ppm 이상인 시간(분), 최고값, 환기 강화가 켜진 시간(에너지 비용 대신)

가정(결과 해석 때 꼭 함께 쓸 것)
  - 환기 강화 = 환기율 BOOST배(기본 2배, 1.5·3배도 같이 계산). 실제 설비 능력은 확인하지 않았다.
  - 창문 열기 같은 사람의 행동은 '사람이 내는 양'에 섞여 그대로 들어간다.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import forecast as F

HOL = "2026-09-24,2026-09-25,2026-09-26,2026-10-05,2026-10-09"
SPLIT = "2026-09-01,2026-10-07,2026-10-14,2026-10-22"
LIMIT, OFF = 1000.0, 900.0
DT = 5 / 60                       # 시간 단위


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="real.csv")
    ap.add_argument("--schedule", default="schedule_real.csv")
    ap.add_argument("--model", default="results_real/gru_main_L24.pt")
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--holidays", default=HOL)
    ap.add_argument("--boost", default="1.5,2,3")
    ap.add_argument("--lead", type=int, default=6, help="예측 규칙이 보는 칸 수 (6 = 30분)")
    ap.add_argument("--predictor", default="gru", choices=["gru", "trend"], help="trend = torch 없이 점검용(최근 15분 기울기 연장)")
    ap.add_argument("--out", default="results_real")
    a = ap.parse_args()
    t = a.split.split(",")
    F.SPLIT = {"train": (t[0], t[1]), "val": (t[1], t[2]), "test": (t[2], t[3])}
    F.HOLIDAYS = set(a.holidays.split(","))
    out = Path(a.out); out.mkdir(exist_ok=True)
    lines = []

    def log(*x):
        s = " ".join(str(i) for i in x); print(s, flush=True); lines.append(s)

    d, sch, flt = F.load(a.csv, a.schedule, None)
    grid, idx, spaces = F.clean_grid(d, flt)
    data = F.Data(grid, idx, spaces, sch)
    t0, t1 = (pd.Timestamp(x, tz=F.TZ) for x in F.SPLIT["test"])
    T = np.where((idx >= t0) & (idx < t1))[0]
    T = T[T < len(idx) - 1]
    log(f"# 환기 제어 규칙 효과 추정 — 시험 주 {t0:%m-%d} ~ {t1 - pd.Timedelta('1s'):%m-%d}, 5분 칸 {len(T)}개 × 공간 {len(spaces)}")

    model = None
    if a.predictor == "gru":
        import torch
        torch.set_num_threads(4)
        hist0, fut0, _ = data.arrays(spaces[0])
        model = F.build("gru", hist0.shape[1], fut0.shape[1])
        model.load_state_dict(torch.load(a.model, map_location="cpu")); model.eval()
        log(f"예측기: GRU ({a.model}), 입력 {hist0.shape[1]}개")
    else:
        log("예측기: 최근 15분 기울기 연장(점검용)")

    # ---- 공간별 물리 값 추정
    phys = {}
    for sp in spaces:
        c = grid[sp].co2
        bg = float(c[(idx < t0)].quantile(0.02))
        dc = (c.shift(-1) - c) / DT                       # ppm/h
        ex = c - bg
        night = (idx.hour >= 22) | (idx.hour < 6)
        m = night & (ex > 40) & dc.notna() & (dc < 0) & (idx < t0)
        lam = float(np.median(-dc[m] / ex[m])) if m.sum() > 30 else 0.3
        phys[sp] = dict(bg=bg, lam=lam)
    log("\n## 방마다 추정한 값 (학습·검증 기간의 밤 데이터)")
    log(pd.DataFrame(phys).T.rename(columns=dict(bg="바깥 수준 ppm", lam="평소 환기율 /시간")).round(3).to_string())

    def predict_max(sp, i, c_sim):
        """칸 i 시점까지의 (가상) 과거로 앞으로 lead 칸 최댓값 예측."""
        if a.predictor == "trend":
            slope = (c_sim[i] - c_sim[i - 3]) / 3
            return c_sim[i] + slope * a.lead
        import torch
        hist, fut, raw = arrays[sp]
        x = hist[i - 23: i + 1].copy()
        x[:, 0] = (c_sim[i - 23: i + 1] - data.mu["co2"]) / data.sd["co2"]
        if np.isnan(x).any():
            return np.nan
        with torch.no_grad():
            dl = model(torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(fut[i + 1: i + 1 + F.H][None])).numpy()[0]
        return float(c_sim[i] + (dl[:a.lead, 0] * data.dsd[0]).max())

    arrays = {sp: data.arrays(sp) for sp in spaces}
    boosts = [float(b) for b in a.boost.split(",")]
    rows, ev_rows = [], []
    for boost in boosts:
        for policy in ("없음", "뒤늦게", "예측"):
            for sp in spaces:
                c_obs = grid[sp].co2.values.astype(float)
                bg, lam = phys[sp]["bg"], phys[sp]["lam"]
                gen = (np.roll(c_obs, -1) - c_obs) / DT + lam * (c_obs - bg)    # 사람이 내는 양(+창문 등), ppm/h
                c_sim = c_obs.copy()
                on = np.zeros(len(c_obs), bool)
                state = False
                for i in T:
                    if np.isnan(c_obs[i]) or np.isnan(c_obs[min(i + 1, len(c_obs) - 1)]) or np.isnan(c_sim[i]):
                        c_sim[i + 1] = c_obs[i + 1]; state = False
                        continue
                    if policy == "뒤늦게":
                        state = c_sim[i] >= LIMIT or (state and c_sim[i] >= OFF)
                    elif policy == "예측":
                        pm = predict_max(sp, i, c_sim)
                        hi = c_sim[i] if np.isnan(pm) else max(c_sim[i], pm)
                        state = hi >= LIMIT or (state and hi >= OFF)
                    on[i] = state
                    lam_t = lam * boost if state else lam
                    c_sim[i + 1] = c_sim[i] + DT * (gen[i] - lam_t * (c_sim[i] - bg))
                seg = c_sim[T]
                obs_seg = c_obs[T]
                rows.append(dict(배수=boost, 규칙=policy, 공간=sp, 초과_분=int(np.nansum(seg >= LIMIT) * 5),
                                 최고_ppm=round(float(np.nanmax(seg))), 평균_ppm=round(float(np.nanmean(seg))),
                                 환기강화_분=int(on[T].sum() * 5), 실제초과_분=int(np.nansum(obs_seg >= LIMIT) * 5)))
                if boost == 2.0 and policy != "없음":
                    pd.DataFrame(dict(ts=idx[T], 실제=obs_seg, 가상=seg, 환기강화=on[T])).to_csv(
                        out / f"control_{policy}_{sp}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(out / "control_results.csv", index=False)

    log("\n## 1,000ppm 이상인 시간 (분, 시험 주 전체)")
    for boost in boosts:
        r = res[res.배수 == boost]
        tab = r.pivot(index="규칙", columns="공간", values="초과_분").reindex(["없음", "뒤늦게", "예측"])
        tab["합계"] = tab.sum(axis=1)
        en = r.pivot(index="규칙", columns="공간", values="환기강화_분").reindex(["없음", "뒤늦게", "예측"]).sum(axis=1)
        tab["환기강화_분(합)"] = en
        base = tab.loc["없음", "합계"]
        tab["초과_감소율"] = ((1 - tab["합계"] / base) * 100).round(1) if base else np.nan
        log(f"\n[환기 강화 = 평소의 {boost:g}배]")
        log(tab.to_string())
    r2 = res[res.배수 == 2.0]
    log("\n## 최고 CO2 (ppm, 환기 2배)")
    log(r2.pivot(index="규칙", columns="공간", values="최고_ppm").reindex(["없음", "뒤늦게", "예측"]).to_string())
    (out / "control_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n끝. 결과: {out}/control_summary.txt, control_results.csv")


if __name__ == "__main__":
    main()
