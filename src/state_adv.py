#!/usr/bin/env python3
"""
과제 5 심화 — ① 처음 보는 공간에서 사용 여부 추정  ② 노쇼·예약 없는 사용 찾기

  ~/collector/venv/bin/python state_adv.py        # 합성(정답 있음) + 실데이터(real.csv 있으면, 후보 목록만)

① 한 공간을 학습에서 통째로 빼고 학습 → 그 공간의 시험 주를 실제 인원(정답)으로 채점.
   공간 이름(원-핫)은 입력에서 뺀다(처음 보는 공간이니 이름을 모름). 비교: 모든 공간으로 학습한 모델, 단순 규칙.
② 사용 여부 분류기(관찰 정답으로 학습, 학습 기간만)를 검증·시험 주에 적용해
   노쇼          : 예약 시작부터 20분 동안 분류기가 계속 '비어있음' → 노쇼로 판정
                   정답 = 예약 시작 후 15분 동안 실제 인원 0 (과제 정의: 예약 후 15분간 비어있음)
                   20분을 보는 이유: 사람이 들어와도 CO2가 오르기까지 5~10분 걸려 처음 칸은 '비어있음'으로 보이기 쉬움
   예약 없는 사용 : 예약(앞뒤 여유 포함) 밖에서 분류기가 15분 이상 연속 '사용중' → 한 건
                   정답 = 같은 구간에서 실제 인원 ≥ 1명이 15분 이상
   비교: 단순 규칙(15분 CO2 +30ppm)으로 같은 판정을 했을 때
③ 실데이터 실습실: 같은 판정으로 '수업인데 안 쓴 것 같은 구간', '수업 없는데 쓴 것 같은 구간' 후보 목록(정답 없음)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import state as S

PRE, POST = pd.Timedelta("10min"), pd.Timedelta("25min")     # 예약 앞뒤 여유(일찍 오기·연장)
NOSHOW_WIN, NOSHOW_TRUE = pd.Timedelta("20min"), pd.Timedelta("15min")
MIN_RUN = 3                                                  # 15분 = 5분 칸 3개


def runs(flag, idx):
    """True 가 MIN_RUN 칸 이상 이어진 구간들 [(시작, 끝)]."""
    out, start, n = [], None, 0
    for t, f in zip(idx, flag):
        if f:
            start = t if start is None else start
            n += 1
        else:
            if start is not None and n >= MIN_RUN:
                out.append((start, prev + pd.Timedelta("5min")))
            start, n = None, 0
        prev = t
    if start is not None and n >= MIN_RUN:
        out.append((start, prev + pd.Timedelta("5min")))
    return out


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def booking_mask(idx, bookings):
    m = np.zeros(len(idx), bool)
    for _, r in bookings.iterrows():
        m |= (idx >= r.start - PRE) & (idx < r.end + POST)
    return m


def detect(df, pred, bookings, truth_occ=None):
    """한 공간의 판정. df=5분 표(시간순), pred=사용중 0/1 배열."""
    idx = df.index
    pred = pd.Series(pred, index=idx)
    ns = []
    for _, r in bookings.iterrows():
        w = pred[(idx >= r.start) & (idx < r.start + NOSHOW_WIN)]
        if len(w) < 3:
            continue
        row = dict(시작=r.start, 끝=r.end, 판정_노쇼=bool((w == 0).all()))
        if truth_occ is not None:
            tw = truth_occ[(truth_occ.index >= r.start) & (truth_occ.index < r.start + NOSHOW_TRUE)]
            row["실제_노쇼"] = bool((tw < 0.5).all()) if len(tw) else np.nan
        ns.append(row)
    out_b = ~booking_mask(idx, bookings)
    ev_p = runs((pred.values == 1) & out_b, idx)
    ev_t = runs((truth_occ.reindex(idx).values >= 1) & out_b, idx) if truth_occ is not None else []
    return pd.DataFrame(ns), ev_p, ev_t


def score_events(ev_p, ev_t):
    hit_p = sum(any(overlap(p, t) for t in ev_t) for p in ev_p)
    hit_t = sum(any(overlap(t, p) for p in ev_p) for t in ev_t)
    return dict(판정건수=len(ev_p), 실제건수=len(ev_t), 맞은판정=hit_p, 찾은실제=hit_t,
                정밀도=round(hit_p / len(ev_p), 3) if ev_p else np.nan, 재현율=round(hit_t / len(ev_t), 3) if ev_t else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="synthetic.csv")
    ap.add_argument("--schedule", default="schedule.csv")
    ap.add_argument("--faults", default="faults.csv")
    ap.add_argument("--truth", default="occupancy_truth.csv")
    ap.add_argument("--real", default="real.csv")
    ap.add_argument("--real_schedule", default="schedule_real.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.metrics import f1_score
    lines = []

    def log(*x):
        s = " ".join(str(i) for i in x); print(s, flush=True); lines.append(s)

    d, sch, flt, occ = S.load(a)
    spaces = sorted(d.space.unique())
    allf = pd.concat([S.frame(d, sch, flt, occ, s) for s in spaces])
    tr, va, te = S.split(allf, S.TRAIN), S.split(allf, S.VAL), S.split(allf, S.TEST)
    log("# 과제 5 심화 (합성 데이터, 정답 = occupancy_truth.csv 실제 인원 ≥ 1)")
    log(f"학습 {S.TRAIN[0]:%m-%d}~{S.TRAIN[1]:%m-%d} / 시험 {S.TEST[0]:%m-%d}~{S.TEST[1]:%m-%d}, 공간 {spaces}\n")

    # ================================================= ① 처음 보는 공간
    log("## ① 한 공간을 학습에서 빼고, 그 공간의 시험 주를 채점 (관찰 정답 기준)")
    rows = []
    for hold in spaces:
        teh = te[te.space == hold]
        y = teh.label_obs.values
        r_rule = S.report(y, (teh.dco2_15 >= 30).astype(int).values)
        res = {"단순 규칙": r_rule}
        for lab, nm in (("label_obs", "관찰 정답"), ("label_sched", "일정 정답")):
            seen = HGB(max_iter=300, random_state=0).fit(tr[S.OCC_X].values, tr[lab])
            tro = tr[tr.space != hold]
            unseen = HGB(max_iter=300, random_state=0).fit(tro[S.OCC_X].values, tro[lab])
            res[f"모든 공간 학습 ({nm})"] = S.report(y, seen.predict(teh[S.OCC_X].values))
            res[f"이 공간 빼고 학습 ({nm})"] = S.report(y, unseen.predict(teh[S.OCC_X].values))
        for k, r in res.items():
            rows.append(dict(시험공간=hold, 방법=k, 정확도=r["정확도"], F1=r["F1"], 재현율=r["재현율"], 빈방을사용중으로=r["빈방을사용중으로"]))
    h = pd.DataFrame(rows)
    h.to_csv(out / "state_holdout.csv", index=False)
    log(h.pivot(index="방법", columns="시험공간", values="F1").to_string())
    log("\n(빈 방을 사용중으로 잘못 본 비율)")
    log(h.pivot(index="방법", columns="시험공간", values="빈방을사용중으로").to_string())

    # ================================================= ② 노쇼·예약 없는 사용
    log("\n## ② 노쇼·예약 없는 사용 찾기 (검증+시험 주, 분류기는 학습 기간만으로 학습)")
    clf = HGB(max_iter=300, random_state=0).fit(np.hstack([tr[S.OCC_X].values, S.onehot(tr, spaces)]), tr.label_obs)
    ev_rows, ns_rows, ev_list = [], [], []
    period = (S.VAL[0], S.TEST[1])
    for sp in sorted(sch.space.unique()):
        f = allf[(allf.space == sp) & (allf.index >= period[0]) & (allf.index < period[1])].dropna(subset=S.OCC_X)
        bk = sch[(sch.space == sp) & (sch.start >= period[0]) & (sch.start < period[1])]
        truth = occ[sp]
        preds = {"분류기": clf.predict(np.hstack([f[S.OCC_X].values, S.onehot(f, spaces)])),
                 "단순 규칙": (f.dco2_15 >= 30).astype(int).values}
        for name, p in preds.items():
            ns, ev_p, ev_t = detect(f, p, bk, truth)
            if len(ns):
                tp = int((ns.판정_노쇼 & ns.실제_노쇼).sum()); fp = int((ns.판정_노쇼 & ~ns.실제_노쇼.astype(bool)).sum())
                fn = int((~ns.판정_노쇼 & ns.실제_노쇼.astype(bool)).sum())
                ns_rows.append(dict(공간=sp, 방법=name, 예약수=len(ns), 실제노쇼=int(ns.실제_노쇼.sum()), 판정노쇼=int(ns.판정_노쇼.sum()),
                                    맞춤=tp, 헛판정=fp, 놓침=fn, 정밀도=round(tp / max(tp + fp, 1), 3), 재현율=round(tp / max(tp + fn, 1), 3)))
                if name == "분류기":
                    ns.assign(공간=sp).to_csv(out / f"state_noshow_{sp}.csv", index=False)
            ev_rows.append(dict(공간=sp, 방법=name, **score_events(ev_p, ev_t)))
            if name == "분류기":
                for p_ in ev_p:
                    ev_list.append(dict(공간=sp, 시작=p_[0], 끝=p_[1], 분=int((p_[1] - p_[0]).total_seconds() // 60),
                                        실제로_사용=any(overlap(p_, t_) for t_ in ev_t)))
    nsdf, evdf = pd.DataFrame(ns_rows), pd.DataFrame(ev_rows)
    nsdf.to_csv(out / "state_noshow_score.csv", index=False); evdf.to_csv(out / "state_unbooked_score.csv", index=False)
    pd.DataFrame(ev_list).to_csv(out / "state_unbooked_events.csv", index=False)
    log("\n[노쇼] 예약 시작 20분 동안 계속 '비어있음' → 노쇼 판정 / 정답: 시작 후 15분 실제 인원 0")
    log(nsdf.to_string(index=False))
    log("\n[예약 없는 사용] 예약(앞 10분·뒤 25분 여유) 밖에서 15분 이상 '사용중' / 정답: 같은 구간 실제 인원 ≥1이 15분 이상")
    log(evdf.to_string(index=False))

    # ================================================= ③ 실데이터 실습실 (정답 없음)
    if Path(a.real).exists() and Path(a.real_schedule).exists():
        log("\n## ③ 실데이터 실습실 — 후보 목록 (정답이 없어 맞았는지는 모름)")
        rd = pd.read_csv(a.real)
        rd["ts"] = pd.to_datetime(rd.ts, utc=True, format="ISO8601").dt.tz_convert(S.TZ)
        rs = pd.read_csv(a.real_schedule)
        for c in ("start", "end"):
            rs[c] = pd.to_datetime(rs[c], utc=True, format="ISO8601").dt.tz_convert(S.TZ)
        rd = rd[rd.space == "실습실"].copy()
        g_idx = pd.date_range(rd.ts.min().floor("5min"), rd.ts.max().floor("5min"), freq="5min")
        no_truth = pd.DataFrame({"실습실": np.nan}, index=g_idx)
        nofault = pd.DataFrame(columns=["space", "metric", "start", "end", "kind"])
        rf = S.frame(rd, rs, nofault, no_truth, "실습실").dropna(subset=S.OCC_X)
        rf["space"] = "강의실A"                                   # 학습 때 가장 비슷한 공간(서비스와 같은 대응)
        p = clf.predict(np.hstack([rf[S.OCC_X].values, S.onehot(rf, spaces)]))
        bk = rs[rs.space == "실습실"]
        ns, ev_p, _ = detect(rf, p, bk, None)
        cand_ns = ns[ns.판정_노쇼] if len(ns) else ns
        log(f"수업 구간 {len(ns)}개 중 시작 20분 동안 계속 '비어있음' {len(cand_ns)}개 (노쇼·휴강 후보):")
        if len(cand_ns):
            log(cand_ns.assign(요일=cand_ns.시작.dt.strftime("%a"), 시작=cand_ns.시작.dt.strftime("%m-%d %H:%M"),
                               끝=cand_ns.끝.dt.strftime("%H:%M"))[["시작", "끝", "요일"]].to_string(index=False))
        ev = pd.DataFrame([dict(시작=s_, 끝=e_, 분=int((e_ - s_).total_seconds() // 60)) for s_, e_ in ev_p])
        if len(ev):
            ev = ev[ev.분 >= 30].copy()
            ev["요일"] = ev.시작.dt.strftime("%a")
            # 같은 날 앞뒤로 수업이 있으면 '수업 사이(점심)' — 남아 있는 사람이 있어도 이상하지 않음
            def kind(r):
                day = bk[bk.start.dt.date == r.시작.date()]
                if (day.end <= r.시작).any() and (day.start >= r.끝 - POST).any():
                    return "수업 사이(점심)"
                return "수업 없는 날" if day.empty else "수업 전·후"
            ev["구분"] = ev.apply(kind, axis=1)
            ev.to_csv(out / "state_real_unbooked.csv", index=False)
            log(f"\n수업 밖에서 30분 이상 '사용중' {len(ev)}건 (예약 없는 사용 후보):")
            log(ev.groupby("구분").분.agg(["count", "sum"]).rename(columns={"count": "건", "sum": "분"}).to_string())
            oth = ev[ev.구분 != "수업 사이(점심)"]
            if len(oth):
                log("\n점심 말고 (수업 없는 날·수업 전후) 날짜별:")
                log(oth.groupby([oth.시작.dt.strftime("%m-%d %a"), "구분"]).분.agg(["count", "sum"]).rename(columns={"count": "건", "sum": "분"}).to_string())
        if len(ns):
            ns.to_csv(out / "state_real_noshow.csv", index=False)
    (out / "state_adv_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n끝. 결과: {out}/state_adv_summary.txt")


if __name__ == "__main__":
    main()
