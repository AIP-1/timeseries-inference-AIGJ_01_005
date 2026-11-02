#!/usr/bin/env python3
"""
과제 2 — 데이터 진단 + 기준선 성능

  python3 analyze.py measurements.csv            # 결과는 out/ 폴더
  python3 analyze.py measurements.csv --interval 5min --out out

입력: 수집기 PostgreSQL 에서 내보낸 CSV (ts, received_at, space, point, device, metric, value, parser)
      합성 데이터도 같은 열 이름이면 그대로 돌아감.
출력: out/diag_missing.csv, diag_outliers.csv, response.csv, baseline.csv, split.csv, *.png, summary.md
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Seoul"

# ---------------------------------------------------------------- 정제 규칙
RANGE = {"co2": (350, 5000), "temperature": (5, 40), "humidity": (5, 100)}   # 실내에서 물리적으로 가능한 범위
SPIKE = {"co2": 300, "temperature": 3.0, "humidity": 15}                      # 한 번 측정 사이 이 이상 변하면 튐
FLAT_MIN = {"co2": 120, "temperature": 360, "humidity": 600}   # 이 분 이상 값이 전혀 안 바뀌면 '멈춘 센서' 의심 (습도는 원래 천천히 변함)
SPATIAL = {"temperature": 3.0, "humidity": 15}                               # 같은 공간 다른 센서들과 이만큼 다르면 위치 확인
SHORT_GAP = {"co2": "10min", "temperature": "30min", "humidity": "30min"}   # 이하 결측은 선형 보간, 넘으면 비워 둠(학습·평가 제외). 인공 결측 실험으로 정함

# 분석 대상: 각 공간의 1분 주기 대표 센서 (센서끼리 보정값이 달라 평균하지 않음)
TARGETS = [
    ("실습실 CO2", "실습실", "co2", "AM107-067999"),
    ("실습실 온도", "실습실", "temperature", "AM107-067999"),
    ("사무실 CO2", "사무실", "co2", "AM103-081175"),
    ("사무실 온도", "사무실", "temperature", "AM103-081175"),
]
# 수업 일정 (월~금)
EVENTS = [("09:00", "수업 시작", +1), ("11:30", "점심 시작", -1), ("13:00", "오후 수업 시작", +1), ("18:00", "수업 끝", -1)]
CLASS = [("09:00", "11:30"), ("13:00", "18:00")]
HORIZONS = [5, 15, 30, 60]   # 분


def load(path):
    d = pd.read_csv(path)
    for c in ("ts", "received_at"):
        d[c] = pd.to_datetime(d[c], utc=True, format="ISO8601").dt.tz_convert(TZ)
    return d


# ---------------------------------------------------------------- 진단
def diag_missing(d):
    rows = []
    env = d[d.metric.isin(RANGE)]
    for (sp, m, dev), g in env.groupby(["space", "metric", "device"]):
        t = g.ts.drop_duplicates().sort_values()
        dt = t.diff().dt.total_seconds().dropna()
        period = float(dt.median()) if len(dt) else float("nan")
        span = (t.iloc[-1] - t.iloc[0]).total_seconds()
        expected = span / period + 1 if period else len(t)
        gaps = dt[dt > period * 3]
        rows.append(dict(공간=sp, 항목=m, 기기=dev, 주기_초=round(period), 건수=len(t),
                         결측률=round(max(0, 1 - len(t) / expected) * 100, 1),
                         긴결측_횟수=len(gaps), 최장결측_분=round(gaps.max() / 60, 1) if len(gaps) else 0))
    return pd.DataFrame(rows)


def diag_outliers(d):
    rows, flags = [], []
    env = d[d.metric.isin(RANGE)].copy()
    env["bad"] = False
    for (sp, m, dev), g in env.groupby(["space", "metric", "device"]):
        g = g.sort_values("ts")
        lo, hi = RANGE[m]
        out_range = (g.value < lo) | (g.value > hi)
        spike = (g.value.diff().abs() > SPIKE[m]) & (g.value.diff(-1).abs() > SPIKE[m])   # 혼자 튀었다 돌아온 점
        env.loc[g.index[out_range | spike], "bad"] = True
        # 멈춤: 값이 FLAT_MIN 분 이상 그대로
        chg = g.value.ne(g.value.shift()).cumsum()
        run = g.groupby(chg).ts.agg(lambda s: (s.iloc[-1] - s.iloc[0]).total_seconds() / 60).max()
        rows.append(dict(공간=sp, 항목=m, 기기=dev, 범위밖=int(out_range.sum()), 튐=int(spike.sum()),
                         최장_같은값_분=round(run)))
        if run >= FLAT_MIN[m]:
            flags.append(f"{sp} {m} {dev}: {run:.0f}분 동안 값이 그대로 (멈춤 의심)")
    # 공간 안 센서끼리 비교
    for (sp, m), g in env[env.metric.isin(SPATIAL)].groupby(["space", "metric"]):
        med = g.groupby("device").value.median()
        if len(med) < 3:
            continue
        ref = med.median()
        for dev, v in med.items():
            if abs(v - ref) > SPATIAL[m]:
                flags.append(f"{sp} {m} {dev}: 중앙값 {v:.1f}, 같은 공간 센서들({ref:.1f})과 {v - ref:+.1f} 차이 → 위치·보정 확인, 공간 대표값에서 제외")
    return pd.DataFrame(rows), env, flags


def hvac_flat(d):
    """냉난방기 실내 온도처럼 범위 규칙에 안 걸리지만 전혀 안 변하는 값."""
    out = []
    for (sp, m), g in d[d.metric.str.contains("temperature") & ~d.metric.isin(RANGE)].groupby(["space", "metric"]):
        if g.value.nunique() == 1:
            out.append(f"{sp} {m}: {len(g):,}건 모두 {g.value.iloc[0]} (변화 없음 → 분석에서 제외)")
    return out


def series(env, space, metric, device, rule):
    """정제된 시계열: 이상값 제거 → rule 간격 평균 → 짧은 결측만 보간."""
    g = env[(env.space == space) & (env.metric == metric) & (env.device == device) & ~env.bad]
    s = g.set_index("ts").value.sort_index().resample(rule).mean()
    limit = int(pd.Timedelta(SHORT_GAP[metric]) / pd.Timedelta(rule))
    return s.interpolate(limit=limit, limit_area="inside") if limit > 0 else s   # 간격이 결측 기준보다 크면 보간 안 함


# ---------------------------------------------------------------- 일정–CO2 반응
def response(env):
    rows = []
    for label, sp, m, dev in TARGETS:
        if m != "co2":
            continue
        s = series(env, sp, m, dev, "1min").rolling(11, center=True, min_periods=6).mean()   # 11분 이동평균(측정 잡음 제거)
        slope = s.diff()  # ppm/분
        days = sorted({t.date() for t in s.dropna().index})
        for day in days:
            if pd.Timestamp(day).weekday() >= 5:
                continue
            for hhmm, name, sign in EVENTS:
                E = pd.Timestamp(f"{day} {hhmm}", tz=TZ)
                w = s[E - pd.Timedelta("30min"): E + pd.Timedelta("90min")]
                if w.isna().mean() > 0.2 or len(w) < 100 or E not in s.index or pd.isna(s.get(E)):
                    continue
                sl = slope[w.index] * sign
                ok = (sl > 1.5).rolling(10).sum() >= 10                 # 10분 연속 1.5ppm/분 이상 같은 방향
                start = ok[ok].index.min()
                start = start - pd.Timedelta("9min") if pd.notna(start) else None
                v0 = s[E]
                after = s[E: E + pd.Timedelta("90min")]
                reach = after[(after - v0) * sign >= 50].index.min()
                rows.append(dict(대상=label, 날짜=str(day), 일정=f"{hhmm} {name}",
                                 반응시작_분=None if start is None else int((start - E).total_seconds() // 60),
                                 변화50ppm_분=None if pd.isna(reach) else int((reach - E).total_seconds() // 60),
                                 일정시각_CO2=round(v0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 분할 + 정규화
def time_split(index, train=0.7, val=0.15, gap="60min"):
    """시간순 분할. 경계마다 예측 지평(60분)만큼 비워서 정답이 다음 구간으로 새지 않게."""
    t0, t1 = index.min(), index.max()
    a = t0 + (t1 - t0) * train
    b = t0 + (t1 - t0) * (train + val)
    g = pd.Timedelta(gap)
    return {"학습": (t0, a), "검증": (a + g, b), "시험": (b + g, t1)}


# ---------------------------------------------------------------- 기준선
def baselines(s, rule):
    step = pd.Timedelta(rule)
    n1h = int(pd.Timedelta("60min") / step)
    preds = {
        "현재값 유지": s,
        "최근 1시간 평균": s.rolling(n1h, min_periods=int(n1h * 0.75)).mean(),
        "어제 같은 시각": None,   # 지평마다 다름(아래)
    }
    return preds, step


def evaluate(s, rule, mask_fn=None):
    preds, step = baselines(s, rule)
    out = []
    # 모든 지평·모든 기준선이 같은 시점 t 에서 평가되도록 공통 마스크를 만든다
    maxh = max(HORIZONS)
    ok = s.notna() & s.shift(-int(pd.Timedelta(f"{maxh}min") / step)).notna() & preds["최근 1시간 평균"].notna()
    day_ok = {}
    for h in HORIZONS:
        k = int(pd.Timedelta(f"{h}min") / step)
        day_ok[h] = s.shift(int(pd.Timedelta("1D") / step) - k).notna()   # y(t+h-24h)
    use_day = all(day_ok[h][ok].sum() >= 12 for h in HORIZONS)
    if use_day:
        for h in HORIZONS:
            ok &= day_ok[h]
    if mask_fn is not None:
        ok &= mask_fn(s.index)
    for h in HORIZONS:
        k = int(pd.Timedelta(f"{h}min") / step)
        y = s.shift(-k)
        cand = {"현재값 유지": preds["현재값 유지"], "최근 1시간 평균": preds["최근 1시간 평균"]}
        if use_day:
            cand["어제 같은 시각"] = s.shift(int(pd.Timedelta("1D") / step) - k)
        for name, p in cand.items():
            e = (y - p)[ok].abs()
            out.append(dict(지평_분=h, 기준선=name, MAE=round(float(e.mean()), 2), 평가점=int(ok.sum())))
        if not use_day:
            out.append(dict(지평_분=h, 기준선="어제 같은 시각", MAE=None, 평가점=0))
    return pd.DataFrame(out)


def in_class(idx):
    hm = idx.strftime("%H:%M")
    wk = idx.weekday < 5
    m = np.zeros(len(idx), bool)
    for a, b in CLASS:
        m |= (hm >= a) & (hm < b)
    return pd.Series(m & wk, index=idx)


# ---------------------------------------------------------------- 그래프
def charts(env, rule, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK KR", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#b9b8b0", "axes.labelcolor": "#4a4a46",
                         "xtick.color": "#4a4a46", "ytick.color": "#4a4a46", "axes.titleweight": "bold",
                         "axes.titlesize": 12, "axes.titlelocation": "left"})
    C1, C2 = "#2a78d6", "#eb6834"

    def bands(ax, idx):
        for day in sorted({t.date() for t in idx}):
            if pd.Timestamp(day).weekday() >= 5:
                continue
            for a, b in CLASS:
                ax.axvspan(pd.Timestamp(f"{day} {a}", tz=TZ), pd.Timestamp(f"{day} {b}", tz=TZ),
                           color="#e7e5dd", lw=0, zorder=0)
        ax.set_xlim(idx.min(), idx.max())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=TZ))
        ax.grid(axis="y", color="#ebe9e2", lw=0.8)

    files = []
    for metric, unit, title in (("co2", "ppm", "실습실 CO2는 시간표를 따르고, 사무실 CO2는 따로 움직인다"),
                                ("temperature", "°C", "온도는 수업 중 1~2°C 오르고 밤사이 천천히 내려간다")):
        fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
        lines = [(l, sp, dev) for l, sp, m, dev in TARGETS if m == metric]
        for (l, sp, dev), c in zip(lines, (C1, C2)):
            s = series(env, sp, metric, dev, rule)
            ax.plot(s.index, s.values, color=c, lw=2, label=l)
        bands(ax, s.index)
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right", ncol=2)
        ax.text(0.0, -0.32, f"회색 띠 = 수업 시간(09:00–11:30, 13:00–18:00). {rule} 평균, 실데이터.",
                transform=ax.transAxes, fontsize=8.5, color="#6b6a64")
        fig.tight_layout()
        p = out / f"timeline_{metric}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    # 평일 시간대별 평균 (공간 2곳 CO2)
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
    for (l, sp, m, dev), c in zip([t for t in TARGETS if t[2] == "co2"], (C1, C2)):
        s = series(env, sp, m, dev, "10min")
        s = s[s.index.weekday < 5]
        hod = s.groupby((s.index.hour + s.index.minute / 60).round(4)).mean().reindex(np.arange(0, 24, 1 / 6).round(4))  # 데이터 없는 시간은 끊어서 그림
        ax.plot(hod.index, hod.values, color=c, lw=2, label=l)
    for a, b in CLASS:
        ha = int(a[:2]) + int(a[3:]) / 60
        hb = int(b[:2]) + int(b[3:]) / 60
        ax.axvspan(ha, hb, color="#e7e5dd", lw=0, zorder=0)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 24)
    ax.set_xlabel("시각")
    ax.set_ylabel("ppm")
    ax.grid(axis="y", color="#ebe9e2", lw=0.8)
    ax.set_title("평일 시간대별 CO2 평균: 실습실은 수업 시작·점심·퇴실에 맞춰 움직인다")
    days = sorted({t.date() for t in s.index if t.weekday() < 5})
    ax.text(0.0, -0.3, f"10분 평균, 평일 {len(days)}일({', '.join(f'{x:%m-%d}' for x in days)})의 같은 시각 평균. 선이 끊긴 곳은 데이터 없음. 실데이터.",
            transform=ax.transAxes, fontsize=8.5, color="#6b6a64")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    p = out / "hourly_co2.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    files.append(p)
    return files


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--interval", default="5min")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    d = load(a.csv)
    miss = diag_missing(d)
    outl, env, flags = diag_outliers(d)
    flags += hvac_flat(d)
    resp = response(env)

    base_all, base_cls, splits = [], [], []
    for label, sp, m, dev in TARGETS:
        s = series(env, sp, m, dev, a.interval)
        for frame, fn, dst in ((base_all, None, "전체"), (base_cls, in_class, "수업 시간")):
            r = evaluate(s, a.interval, fn)
            r.insert(0, "대상", label)
            r.insert(1, "구간", dst)
            frame.append(r)
        sp_ = time_split(s.dropna().index)
        tr = s[sp_["학습"][0]: sp_["학습"][1]]
        splits.append(dict(대상=label, **{k: f"{v[0]:%m-%d %H:%M} ~ {v[1]:%m-%d %H:%M}" for k, v in sp_.items()},
                           학습_평균=round(tr.mean(), 2), 학습_표준편차=round(tr.std(), 2)))
    base = pd.concat(base_all + base_cls)

    miss.to_csv(out / "diag_missing.csv", index=False)
    outl.to_csv(out / "diag_outliers.csv", index=False)
    resp.to_csv(out / "response.csv", index=False)
    base.to_csv(out / "baseline.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "split.csv", index=False)
    files = charts(env, a.interval, out)

    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# 진단 요약\n\n기간: {d.ts.min():%Y-%m-%d %H:%M} ~ {d.ts.max():%Y-%m-%d %H:%M} ({(d.ts.max()-d.ts.min()).total_seconds()/3600:.1f}시간), "
                f"분석 간격 {a.interval}\n\n## 확인할 센서\n" + "\n".join(f"- {x}" for x in flags) +
                "\n\n## 결측\n" + miss.to_markdown(index=False) + "\n\n## 이상값\n" + outl.to_markdown(index=False) +
                "\n\n## 일정–CO2 반응\n" + resp.to_markdown(index=False) +
                "\n\n## 기준선 MAE\n" + base.to_markdown(index=False) +
                "\n\n## 분할(시간순) + 학습 구간 정규화 값\n" + pd.DataFrame(splits).to_markdown(index=False) + "\n")
    print((out / "summary.md").read_text(encoding="utf-8"))
    print("그래프:", ", ".join(str(p) for p in files))


if __name__ == "__main__":
    main()
