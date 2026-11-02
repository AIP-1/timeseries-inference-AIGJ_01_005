#!/usr/bin/env python3
"""
합성 데이터 생성기 — 실습실 실데이터(2026-10-20~21)의 움직임에 맞춘 4주치 교육장 데이터

  python3 gen_synthetic.py            # → synthetic.csv, schedule.csv, faults.csv

공간 4곳 (과제 자료: 강의실 2, 회의실 1, 라운지 1)
  강의실A  실습실과 같은 시간표 (월~금 09:00-11:30, 13:00-18:00)
  강의실B  월·수·금 10:00-12:00, 화·목 14:00-17:00 (10% 휴강)
  회의실   예약제. 예약 안 지킴(노쇼 20%), 늦게 끝남(25%), 예약 없이 사용
  라운지   일정 없음. 점심·쉬는 시간에 붐빔

물리 모델 (1분 단위)
  CO2:  dC/dt = 인원 x 사람당 배출 / 부피 - 환기율 x (C - 배경값)
  온도: 난방(근무시간) + 바깥 기온 + 사람 열
  센서: 잡음, 2분마다 갱신(1분마다 같은 값 반복), 정수 반올림
마지막 1주에는 고장 5종 x 3건(튄값·값고정·영점틀어짐·냉방정지·환기정지)을 심고 faults.csv 에 정답을 남긴다(과제 4용).
출력 열 이름은 수집기 CSV 와 같다: ts, received_at, space, point, device, metric, value, parser
"""
import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
START = pd.Timestamp("2026-09-21 00:00", tz=TZ)      # 월요일
DAYS = 28
HOLIDAYS = {pd.Timestamp("2026-10-09").date()}        # 한글날(금)
rng = np.random.default_rng(42)

SPACES = {
    #        부피m3  환기(켜짐/꺼짐, 회/시)  배경ppm  난방목표  사람열
    "강의실A": dict(V=300, lam_on=2.3, lam_off=0.55, bg=545, tset=23.6, heat=0.035),
    "강의실B": dict(V=260, lam_on=2.0, lam_off=0.55, bg=520, tset=23.2, heat=0.035),
    "회의실":  dict(V=70,  lam_on=1.6, lam_off=0.40, bg=530, tset=22.8, heat=0.07),
    "라운지":  dict(V=220, lam_on=2.6, lam_off=0.55, bg=500, tset=22.5, heat=0.02),
}
EMIT = 18.7e-3   # 사람당 CO2 m3/h


def minutes():
    return pd.date_range(START, periods=DAYS * 1440, freq="1min")


def is_workday(d):
    return d.weekday() < 5 and d not in HOLIDAYS


def add_block(occ, idx, start, end, n, early=0, late=0):
    """[start-early, end+late) 동안 n명. 들어오고 나갈 때 몇 분에 걸쳐 바뀜."""
    a = idx.searchsorted(start - pd.Timedelta(minutes=early))
    b = idx.searchsorted(end + pd.Timedelta(minutes=late))
    ramp = 6
    prof = np.full(b - a, float(n))
    r = min(ramp, len(prof) // 2)
    if r:
        prof[:r] *= np.linspace(0.3, 1, r)
        prof[-r:] *= np.linspace(1, 0.2, r)
    occ[a:b] += prof


def schedules(idx):
    occ = {s: np.zeros(len(idx)) for s in SPACES}
    plan = []       # 미리 알 수 있는 일정 (모델 입력)
    for day in sorted({t.date() for t in idx}):
        D = lambda hm: pd.Timestamp(f"{day} {hm}", tz=TZ)
        wd = pd.Timestamp(day).weekday()
        work = is_workday(day)
        # 강의실A
        if work:
            n = rng.integers(14, 21)
            early = int(rng.integers(0, 16))
            end_pm = D("17:00") if rng.random() < 0.1 else D("18:00")      # 가끔 일찍 끝남
            plan += [("강의실A", D("09:00"), D("11:30"), "수업"), ("강의실A", D("13:00"), D("18:00"), "수업")]
            add_block(occ["강의실A"], idx, D("09:00"), D("11:30"), n, early=early, late=int(rng.integers(0, 6)))
            add_block(occ["강의실A"], idx, D("11:30"), D("13:00"), n * rng.uniform(0.05, 0.25))   # 점심에 남는 사람
            add_block(occ["강의실A"], idx, D("13:00"), end_pm, n * rng.uniform(0.85, 1.0), early=int(rng.integers(0, 8)), late=int(rng.integers(0, 10)))
        # 강의실B
        if work:
            slot = ("10:00", "12:00") if wd in (0, 2, 4) else ("14:00", "17:00")
            plan.append(("강의실B", D(slot[0]), D(slot[1]), "수업"))
            if rng.random() > 0.1:                                              # 10% 휴강(일정표엔 있음)
                add_block(occ["강의실B"], idx, D(slot[0]), D(slot[1]), rng.integers(12, 21), early=int(rng.integers(0, 10)), late=int(rng.integers(0, 6)))
        # 회의실
        if work:
            t = D("09:00")
            for _ in range(rng.integers(0, 5)):
                t = t + pd.Timedelta(minutes=int(rng.integers(0, 120)))
                dur = int(rng.choice([30, 60, 60, 90, 120]))
                end = t + pd.Timedelta(minutes=dur)
                if end > D("18:30"):
                    break
                plan.append(("회의실", t, end, "예약"))
                if rng.random() > 0.2:                                          # 20% 노쇼
                    over = int(rng.integers(5, 21)) if rng.random() < 0.25 else 0
                    add_block(occ["회의실"], idx, t, end, rng.integers(3, 11), early=int(rng.integers(0, 6)), late=over)
                t = end
            if rng.random() < 0.3:                                              # 예약 없이 사용
                s = D("10:00") + pd.Timedelta(minutes=int(rng.integers(0, 420)))
                add_block(occ["회의실"], idx, s, s + pd.Timedelta(minutes=int(rng.integers(20, 50))), rng.integers(2, 6))
        # 라운지 (일정 없음)
        if work:
            add_block(occ["라운지"], idx, D("11:30"), D("13:20"), rng.integers(8, 20))
            for brk in ("10:50", "15:50"):
                add_block(occ["라운지"], idx, D(brk), D(brk) + pd.Timedelta(minutes=15), rng.integers(4, 12))
        for _ in range(rng.integers(2, 8)):                                     # 오가는 사람
            s = D("08:00") + pd.Timedelta(minutes=int(rng.integers(0, 660)))
            add_block(occ["라운지"], idx, s, s + pd.Timedelta(minutes=int(rng.integers(10, 40))), rng.integers(1, 4))
    return occ, pd.DataFrame(plan, columns=["space", "start", "end", "kind"])


def windows(idx, occ):
    """창문 개방: 강의실에서 주 2~3번, 쉬는 시간 10~20분 동안 환기가 크게 늘어남."""
    w = {s: np.zeros(len(idx)) for s in SPACES}
    ev = []
    for s in ("강의실A", "강의실B"):
        busy = np.where(occ[s] > 5)[0]
        for _ in range(int(DAYS / 7 * rng.uniform(2, 3.5))):
            i = int(rng.choice(busy))
            dur = int(rng.integers(10, 21))
            w[s][i:i + dur] = rng.uniform(4, 8)
            ev.append((s, idx[i], idx[min(i + dur, len(idx) - 1)], "창문 개방"))
    return w, ev


def simulate(idx, occ, win, vent_off=None, cool_off=None):
    hours = idx.hour + idx.minute / 60
    doy = (idx - START).total_seconds() / 86400
    tout = 15 - 0.15 * doy + 5 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.3, len(idx)).cumsum() * 0.02
    on = np.array([(is_workday(t.date()) and 8 <= t.hour < 18) for t in idx])   # 환기·난방 08~18시
    out = {}
    for s, p in SPACES.items():
        C = np.empty(len(idx)); T = np.empty(len(idx)); H = np.empty(len(idx))
        c, temp = p["bg"] + 20, 22.6
        for i in range(len(idx)):
            v_off = vent_off is not None and vent_off[s][i]
            c_off = cool_off is not None and cool_off[s][i]
            base = p["lam_on"] if on[i] else p["lam_off"]
            lam = (base / 1.8 if v_off else base) + win[s][i]          # 환기 정지: 환기량이 줄어 CO2 상승 폭이 약 1.8배
            gen = occ[s][i] * EMIT / p["V"] * 1e6                     # ppm/h
            c += (gen - lam * (c - p["bg"])) / 60
            heat = 0.9 * (p["tset"] - temp) if (on[i] and not c_off) else 0.0
            if c_off:
                heat += 0.6                                                  # 냉방 정지: 조절이 멈추고 기기·햇빛 열이 쌓임
            temp += (heat + 0.012 * (tout[i] - temp) + p["heat"] * occ[s][i] - 0.6 * win[s][i] * 0.05) / 60
            C[i], T[i] = c, temp
            H[i] = 58 + 0.25 * occ[s][i] - 0.8 * (temp - 22.5)
        out[s] = dict(co2=C, temperature=T, humidity=H)
    return out


def sensors(idx, truth):
    """2분마다 갱신, 1분마다 보고. 잡음과 눈금 반올림."""
    rows = []
    for s, m in truth.items():
        dev = f"SYN-{s}"
        for metric, val in m.items():
            noise = {"co2": 8, "temperature": 0.05, "humidity": 0.6}[metric]
            v = val + rng.normal(0, noise, len(val))
            v[1::2] = v[0::2][: len(v[1::2])]                       # 2분 갱신
            v = np.round(v) if metric == "co2" else np.round(v * (10 if metric == "temperature" else 2)) / (10 if metric == "temperature" else 2)
            rows.append(pd.DataFrame(dict(ts=idx, space=s, device=dev, metric=metric, value=v)))
    return pd.concat(rows, ignore_index=True)


def plan_faults(idx, occ):
    """마지막 1주(10-12~10-16)에 고장 5종 x 3건 = 15건. 앞 3주를 그대로 두려고 난수를 따로 씀."""
    T = lambda x: pd.Timestamp(x, tz=TZ)
    ev = [
        # 종류,      공간,     항목,          시작,               끝
        ("튄값",     "강의실A", "co2",         "2026-10-12 10:17", "2026-10-12 10:18"),
        ("튄값",     "회의실",  "temperature", "2026-10-14 15:03", "2026-10-14 15:04"),
        ("튄값",     "라운지",  "co2",         "2026-10-16 12:41", "2026-10-16 12:42"),
        ("값고정",   "강의실B", "co2",         "2026-10-13 13:00", "2026-10-13 16:00"),
        ("값고정",   "라운지",  "temperature", "2026-10-15 09:00", "2026-10-15 12:00"),
        ("값고정",   "강의실A", "co2",         "2026-10-16 13:30", "2026-10-16 15:00"),
        ("영점틀어짐", "라운지",  "co2",         "2026-10-12 20:00", "2026-10-13 08:00"),
        ("영점틀어짐", "강의실B", "co2",         "2026-10-14 00:00", "2026-10-14 12:00"),
        ("영점틀어짐", "회의실",  "co2",         "2026-10-15 18:00", "2026-10-16 06:00"),
        ("냉방정지", "강의실A", "temperature", "2026-10-13 13:00", "2026-10-13 18:00"),
        ("냉방정지", "강의실B", "temperature", "2026-10-15 14:00", "2026-10-15 17:00"),
        ("냉방정지", "강의실A", "temperature", "2026-10-16 09:00", "2026-10-16 11:30"),
        ("환기정지", "강의실A", "co2",         "2026-10-14 09:00", "2026-10-14 11:30"),
        ("환기정지", "강의실B", "co2",         "2026-10-12 10:00", "2026-10-12 12:00"),
        ("환기정지", "회의실",  None,          "2026-10-13 00:00", "2026-10-13 00:00"),   # 실제 회의 시간에 맞춤(아래)
    ]
    out = []
    for kind, sp, m, a, b in ev:
        a, b = T(a), T(b)
        if kind == "환기정지" and sp == "회의실":       # 그날 처음으로 3명 이상 모인 회의 동안
            day = (idx >= T("2026-10-13 08:00")) & (idx < T("2026-10-13 19:00")) & (occ["회의실"] >= 3)
            i = np.where(day)[0]
            j = i[: (np.where(np.diff(i) > 1)[0].min() + 1) if (np.diff(i) > 1).any() else len(i)]
            a, b, m = idx[j[0]], idx[j[-1]] + pd.Timedelta(minutes=1), "co2"
        out.append((sp, m, a, b, kind))
    return pd.DataFrame(out, columns=["space", "metric", "start", "end", "kind"])


def equipment_masks(idx, faults):
    vent = {s: np.zeros(len(idx), bool) for s in SPACES}
    cool = {s: np.zeros(len(idx), bool) for s in SPACES}
    for _, f in faults.iterrows():
        m = (idx >= f.start) & (idx < f.end)
        if f.kind == "환기정지":
            vent[f.space] |= m
        elif f.kind == "냉방정지":
            cool[f.space] |= m
    return vent, cool


def apply_sensor_faults(df, faults):
    r2 = np.random.default_rng(7)
    for _, f in faults.iterrows():
        m = (df.space == f.space) & (df.metric == f.metric) & (df.ts >= f.start) & (df.ts < f.end)
        if f.kind == "튄값":
            df.loc[m, "value"] += 1500 if f.metric == "co2" else 9.0
        elif f.kind == "값고정":
            df.loc[m, "value"] = df.loc[m, "value"].iloc[0]
        elif f.kind == "영점틀어짐":
            n = int(m.sum()); ramp = min(n, 6 * 60)                       # 6시간에 걸쳐 오른 뒤 유지
            add = np.concatenate([np.linspace(0, r2.uniform(250, 320), ramp), np.full(n - ramp, np.nan)])
            add[ramp:] = add[ramp - 1]
            df.loc[m, "value"] += np.round(add)
    return df


def main():
    idx = minutes()
    occ, plan = schedules(idx)
    win, wev = windows(idx, occ)
    faults = plan_faults(idx, occ)
    vent, cool = equipment_masks(idx, faults)
    truth = simulate(idx, occ, win, vent, cool)
    df = sensors(idx, truth)
    df = apply_sensor_faults(df, faults)
    df["received_at"] = df.ts + pd.to_timedelta(rng.uniform(0.1, 0.4, len(df)), unit="s")
    df["point"] = "-"
    df["parser"] = "synthetic"
    df = df[["ts", "received_at", "space", "point", "device", "metric", "value", "parser"]].sort_values(["ts", "space", "metric"])
    df.to_csv("synthetic.csv", index=False)
    plan.to_csv("schedule.csv", index=False)
    faults.to_csv("faults.csv", index=False)
    pd.DataFrame(wev, columns=["space", "start", "end", "kind"]).to_csv("windows.csv", index=False)
    occ_df = pd.DataFrame({s: v for s, v in occ.items()}, index=idx)
    occ_df.resample("5min").mean().round(2).to_csv("occupancy_truth.csv")   # 정답 인원(과제 5용, 모델 입력 아님)
    print(f"synthetic.csv {len(df):,}행, {idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}, 일정 {len(plan)}건, 고장 {faults.kind.nunique()}종 {len(faults)}건, 창문 {len(wev)}번")


if __name__ == "__main__":
    main()
