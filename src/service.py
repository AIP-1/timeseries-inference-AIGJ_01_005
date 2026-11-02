#!/usr/bin/env python3
"""
과제 8 — 상시 추론 서비스 v1

매분 수집기 DB(~/collector/data/collector.db)를 읽어 공간마다 계산하고 내보낸다.
  ① 60분 예측(CO2·온도, 5분 x 12칸) + "몇 분 뒤 1,000ppm 초과" 경보
  ② 이상 여부·이상 점수·원인 센서 + 규칙 경보(데이터 끊김·값 고정·범위 밖·튐)
  ③ 사용 여부 + 환기 선제 경보(30분 안에 1,000ppm)
내보내기
  MQTT  : team2/service/<공간>  (retain → 새 구독자도 최신값을 바로 받음)
          team2/service/<공간>/alert  (경보가 새로 생기거나 사라질 때만, 진행 중이면 30분마다 다시)
  조회 API: http://<보드>:8080/state/<공간>, /forecast/<공간>?minutes=30, /spaces, /selfcheck, /health
          스스로 채점: 30·60분 뒤 실제 값이 들어오면 채점해 결과의 self_check 에 싣고, 나빠지면 '예측성능저하' 경보
전처리는 학습과 같다(과제 2 규칙): 범위·튐 제거 → 다 찬 5분 칸만 평균 → CO2 10분·온습도 30분까지만 보간.

  ~/collector/venv/bin/python service.py            # 상시 실행 (systemd 로 등록)
  ~/collector/venv/bin/python service.py --once     # 한 번만 계산하고 결과 출력 (점검용)
"""
import argparse
import json
import logging
import math
import os
import pickle
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
TZ = "Asia/Seoul"
HOME = Path.home()
COLLECTOR = HOME / "collector"

# 실제 공간 → 대표 센서, 학습 때의 공간(합성 데이터 이름), 일정
SPACES = {
    "실습실": dict(alias="lab", device="AM107-067999", model_space="강의실A", timetable=True),
    "사무실": dict(alias="office", device="AM103-081175", model_space="라운지", timetable=False),
    "회의실": dict(alias="meeting", device="AM103-089152", model_space="회의실", timetable=False),
}
TIMETABLE = [("09:00", "11:30"), ("13:00", "18:00")]       # 실습실 월~금
HOLIDAYS = {"2026-10-09"}
METRICS = ["co2", "temperature", "humidity"]
RANGE = {"co2": (350, 5000), "temperature": (5, 40), "humidity": (5, 100)}
SPIKE = {"co2": 300, "temperature": 3.0, "humidity": 15}
STUCK_MIN = {"co2": 120, "temperature": 360, "humidity": 600}
GAP = {"co2": 2, "temperature": 6, "humidity": 6}            # 보간할 최대 5분 칸
LIMIT = 1000
H = 12
TOPIC = "team2/service"
TYPICAL_ERR = {"co2": 12.3, "temperature": 0.044}           # 과제 3 GRU 30분 MAE → 이상 점수 눈금
ANOM_THR = 6.0
# 스스로 채점(과제 8 심화): 예측을 내놓은 뒤 실제 값이 들어오면 채점한다
SC_HOURS = 24                      # 성적을 모으는 기간
SC_RECENT = 24                     # '최근' = 마지막 24번 채점 (60분 지평이면 2시간)
SC_MIN_N = 12                      # 이보다 적게 채점됐으면 판정하지 않음
SC_WORSE = 1.10                    # 최근 오차가 '현재값 유지'보다 10% 넘게 크면 경보
SC_JUMP = 2.0                      # 최근 오차가 지난 24시간 오차의 2배를 넘으면 경보
log = logging.getLogger("service")


# ------------------------------------------------------------------ 모델 준비 (학습 때 쓴 값 그대로)
class Models:
    def __init__(self):
        self.gru = self.norm = self.occ = self.vent = None
        self.notes = []
        self._load_forecast()
        self._load_classifiers()

    def _load_forecast(self):
        try:
            import torch
            sys.path.insert(0, str(BASE))
            import forecast as F
            cache = BASE / "service_norm.json"
            if cache.exists():
                self.norm = json.loads(cache.read_text())
            else:
                d, sch, flt = F.load(BASE / "synthetic.csv", BASE / "schedule.csv", BASE / "faults.csv")
                grid, idx, spaces = F.clean_grid(d, flt)
                data = F.Data(grid, idx, spaces, sch)
                self.norm = dict(mu=data.mu[F.INPUTS].tolist(), sd=data.sd[F.INPUTS].tolist(), dsd=data.dsd.tolist(), spaces=spaces)
                cache.write_text(json.dumps(self.norm, ensure_ascii=False))
            ckpt = BASE / "results" / "gru_main_L24.pt"
            n_in = 3 + 3 + 1 + len(self.norm["spaces"])
            m = F.build("gru", n_in, 4)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.eval()
            torch.set_num_threads(2)
            self.gru, self.torch = m, torch
            self.notes.append("예측: GRU (과제 3)")
        except Exception as e:
            self.notes.append(f"예측: 현재값 유지로 대체 ({type(e).__name__}: {e})")

    def _load_classifiers(self):
        try:
            cache = BASE / "service_classifiers.pkl"
            if cache.exists():
                self.occ, self.vent, self.cls_spaces, self.vent_thr = pickle.loads(cache.read_bytes())
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                sys.path.insert(0, str(BASE))
                import state as S
                ns = argparse.Namespace(csv=BASE / "synthetic.csv", schedule=BASE / "schedule.csv", faults=BASE / "faults.csv",
                                        truth=BASE / "occupancy_truth.csv")
                d, sch, flt, occ = S.load(ns)
                spaces = sorted(d.space.unique())
                allf = pd.concat([S.frame(d, sch, flt, occ, s) for s in spaces])
                tr = S.split(allf, S.TRAIN)
                X = np.hstack([tr[S.OCC_X].values, S.onehot(tr, spaces)])
                self.occ = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, tr.label_sched)
                v = tr[tr.vent_valid]
                w = np.where(v.vent_pos == 1, (1 - v.vent_pos.mean()) / v.vent_pos.mean(), 1.0)
                Xv = np.hstack([v[S.VENT_X].values, S.onehot(v, spaces)])
                self.vent = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xv, v.vent_pos, sample_weight=w)
                self.cls_spaces, self.vent_thr = spaces, 0.5
                cache.write_bytes(pickle.dumps((self.occ, self.vent, self.cls_spaces, self.vent_thr)))
            self.notes.append("사용 여부·환기 경보: 분류기 (과제 5)")
        except Exception as e:
            self.notes.append(f"사용 여부·환기 경보: 단순 규칙으로 대체 ({type(e).__name__}: {e})")


# ------------------------------------------------------------------ 데이터 읽기 + 전처리
def read_raw(con, device, hours=13):
    since = time.time() - hours * 3600
    rows = con.execute("SELECT ts, metric, value FROM measurements WHERE device=? AND ts>=? AND metric IN ('co2','temperature','humidity')",
                       (device, since)).fetchall()
    if not rows:
        return pd.DataFrame(columns=METRICS)
    df = pd.DataFrame(rows, columns=["ts", "metric", "value"])
    df["ts"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert(TZ)
    return df.pivot_table(index="ts", columns="metric", values="value").reindex(columns=METRICS).sort_index()


def clean(raw):
    """과제 2·3과 같은 규칙. 범위 밖·혼자 튄 값을 지운 1분 표."""
    out = raw.copy()
    for m in METRICS:
        v = out[m]
        lo, hi = RANGE[m]
        bad = (v < lo) | (v > hi) | ((v.diff().abs() > SPIKE[m]) & (v.diff(-1).abs() > SPIKE[m]))
        out.loc[bad, m] = np.nan
    return out


def grid5(clean1, now):
    """다 찬 5분 칸만 쓴다. 지금 진행 중인 칸(1~4분치)은 평균이 흔들려서 뺀다."""
    end = now.floor("5min")
    g = clean1[clean1.index < end].resample("5min").mean()
    g = g.reindex(pd.date_range(end - pd.Timedelta("13h"), end - pd.Timedelta("5min"), freq="5min"))
    for m in METRICS:
        g[m] = g[m].interpolate(limit=GAP[m], limit_area="inside")
        g[m] = g[m].ffill(limit=GAP[m])          # 10분 주기 센서: 다음 값이 오기 전까지 마지막 값을 이어 씀(보간 기준 이내)
    return g


def busy_flags(idx, timetable):
    f = np.zeros(len(idx), np.float32)
    if not timetable:
        return f
    hm = idx.strftime("%H:%M")
    work = np.array([t.weekday() < 5 and t.strftime("%Y-%m-%d") not in HOLIDAYS for t in idx])
    for a, b in TIMETABLE:
        f[(hm >= a) & (hm < b) & work] = 1
    return f


def calendar(idx):
    h = (idx.hour + idx.minute / 60) / 24 * 2 * np.pi
    work = np.array([(t.weekday() < 5 and t.strftime("%Y-%m-%d") not in HOLIDAYS) for t in idx], np.float32)
    return np.stack([np.sin(h), np.cos(h), work], 1).astype(np.float32)


# ------------------------------------------------------------------ 규칙 경보
def rule_alerts(raw, clean1, now, period_s):
    alerts = []
    last = raw.index.max() if len(raw) else None
    age = (now - last).total_seconds() if last is not None else float("inf")
    if age > max(600, 2.5 * period_s):
        alerts.append(dict(type="데이터끊김", sensor="*", detail=f"마지막 수신 {int(age // 60)}분 전"))
        return alerts, age
    for m in METRICS:
        v = raw[m].dropna()
        if v.empty:
            continue
        lo, hi = RANGE[m]
        if not (lo <= v.iloc[-1] <= hi):
            alerts.append(dict(type="범위밖", sensor=m, detail=f"{v.iloc[-1]}"))
        grp = v.ne(v.shift()).cumsum()
        run = (v.index[-1] - v[grp == grp.iloc[-1]].index[0]).total_seconds() / 60
        if run >= STUCK_MIN[m]:
            alerts.append(dict(type="값고정", sensor=m, detail=f"{run:.0f}분 동안 {v.iloc[-1]}"))
        recent = v[v.index >= now - pd.Timedelta("15min")]
        med = v[(v.index < recent.index.min())].tail(10).median() if len(recent) else np.nan
        if len(recent) and not np.isnan(med) and (recent - med).abs().max() > SPIKE[m] * 1.3:
            alerts.append(dict(type="튐", sensor=m, detail=f"최근 15분 최대 {recent.max()} (직전 중앙값 {med:.0f})"))
    return alerts, age


# ------------------------------------------------------------------ 공간 하나 계산
class Service:
    def __init__(self, db, models):
        self.db, self.m = db, models
        self.latest = {}                       # 공간 → 최신 결과
        self.past_fc = {s: {} for s in SPACES}  # 공간 → {예측 대상 시각: (co2, temp)} (이상 점수용)
        self.alert_state = {s: {} for s in SPACES}
        self.sc_pending = {s: {} for s in SPACES}  # 공간 → {대상 칸 시각: [(지평 분, 모델 예측, 현재값 유지 예측)]}
        self.sc_scores = {s: [] for s in SPACES}   # 공간 → [(대상 칸 시각, 지평 분, 모델 오차, 현재값 유지 오차)]
        self.sc_file = BASE / "logs" / "selfcheck.jsonl"
        self._sc_reload()

    def _sc_reload(self):
        """재시작해도 지난 24시간 성적이 이어지도록 기록 파일에서 다시 읽는다."""
        try:
            since = pd.Timestamp.now(tz=TZ) - pd.Timedelta(hours=SC_HOURS)
            for line in self.sc_file.read_text(encoding="utf-8").splitlines()[-20000:]:
                r = json.loads(line)
                t = pd.Timestamp(r["target"])
                if t >= since and r["space"] in self.sc_scores:
                    self.sc_scores[r["space"]].append((t, r["h"], r["err"], r["err_persist"]))
        except FileNotFoundError:
            pass
        except Exception as e:
            log.error("스스로 채점 기록 읽기 실패: %s", e)

    def self_check(self, space, g, fc):
        """① 지금 막 다 찬 칸에 대해 예전에 내놓은 예측들을 채점 ② 이번 예측을 채점 대기열에 넣기 ③ 성적 요약."""
        t_now, cur = g.index[-1], g.co2.iloc[-1]
        new = []
        if not pd.isna(cur):
            for h, pred, pers in self.sc_pending[space].pop(t_now, []):
                rec = (t_now, h, abs(pred - cur), abs(pers - cur))
                self.sc_scores[space].append(rec)
                new.append(dict(space=space, target=t_now.isoformat(), h=h, pred=round(pred, 1), actual=round(float(cur), 1),
                                err=round(rec[2], 1), err_persist=round(rec[3], 1)))
        if new:
            try:
                with open(self.sc_file, "a", encoding="utf-8") as f:
                    for r in new:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception as e:
                log.error("스스로 채점 기록 실패: %s", e)
        # 이번 예측 등록 (같은 칸에 대해 매분 다시 계산하므로 칸마다 한 번만)
        if fc.get("method") == "GRU":
            for k in (6, 12):                                   # 30분·60분 뒤만 채점
                tgt = t_now + pd.Timedelta(minutes=5 * k)
                lst = self.sc_pending[space].setdefault(tgt, [])
                if not any(h == 5 * k for h, _, _ in lst):
                    lst.append((5 * k, float(fc["co2"][k - 1]), float(cur)))
        old = t_now - pd.Timedelta(hours=SC_HOURS)
        self.sc_pending[space] = {k: v for k, v in self.sc_pending[space].items() if k > t_now - pd.Timedelta("10min")}
        self.sc_scores[space] = [r for r in self.sc_scores[space] if r[0] >= old]
        return self.sc_summary(space)

    def sc_summary(self, space):
        out = {}
        warn = []
        for h in (30, 60):
            rs = [r for r in self.sc_scores[space] if r[1] == h]
            if not rs:
                continue
            day_m, day_p = float(np.mean([r[2] for r in rs])), float(np.mean([r[3] for r in rs]))
            rec = rs[-SC_RECENT:]
            rec_m, rec_p = float(np.mean([r[2] for r in rec])), float(np.mean([r[3] for r in rec]))
            out[f"h{h}"] = dict(n_24h=len(rs), mae_24h=round(day_m, 1), persist_mae_24h=round(day_p, 1),
                                n_recent=len(rec), mae_recent=round(rec_m, 1), persist_mae_recent=round(rec_p, 1))
            if h == 60 and len(rec) >= SC_MIN_N:
                if rec_m > SC_WORSE * rec_p and rec_m - rec_p > 5:
                    warn.append(f"최근 {len(rec)}번 60분 예측 오차 {rec_m:.0f}ppm > 현재값 유지 {rec_p:.0f}ppm")
                if len(rs) >= 3 * SC_RECENT and rec_m > SC_JUMP * day_m:
                    warn.append(f"최근 60분 예측 오차 {rec_m:.0f}ppm 가 지난 24시간 평균 {day_m:.0f}ppm 의 {rec_m/day_m:.1f}배")
        return dict(unit="ppm (CO2 절대 오차 평균)", scores=out, degraded=bool(warn), reasons=warn,
                    method="예측 30·60분 뒤 실제 칸이 다 차면 채점, 같은 시점 '현재값 유지'와 비교")

    def forecast(self, g, cfg):
        last = g.iloc[-1]
        persist = dict(co2=[float(last.co2)] * H, temperature=[float(last.temperature)] * H, method="현재값 유지")
        if self.m.gru is None:
            return persist
        hist = g.iloc[-24:]
        if hist[METRICS].isna().any().any():
            persist["method"] = "현재값 유지 (최근 2시간 데이터 부족)"
            return persist
        n = self.m.norm
        z = (hist[METRICS].values - np.array(n["mu"])) / np.array(n["sd"])
        idx = hist.index
        oh = np.zeros((len(idx), len(n["spaces"])), np.float32)
        oh[:, n["spaces"].index(cfg["model_space"])] = 1
        x = np.concatenate([z.astype(np.float32), calendar(idx), busy_flags(idx, cfg["timetable"])[:, None], oh], 1)
        fidx = pd.date_range(idx[-1] + pd.Timedelta("5min"), periods=H, freq="5min")
        f = np.concatenate([calendar(fidx), busy_flags(fidx, cfg["timetable"])[:, None]], 1)
        t = self.m.torch
        with t.no_grad():
            dlt = self.m.gru(t.from_numpy(x[None]), t.from_numpy(f[None])).numpy()[0]
        pred = last[["co2", "temperature"]].values + dlt * np.array(n["dsd"])
        return dict(co2=[round(float(v), 1) for v in pred[:, 0]], temperature=[round(float(v), 2) for v in pred[:, 1]], method="GRU")

    def classifier_feats(self, g, cfg):
        f = pd.DataFrame(index=g.index)
        f["co2"], f["temp"], f["hum"] = g.co2, g.temperature, g.humidity
        for k, name in ((1, "5"), (3, "15"), (6, "30"), (12, "60")):
            f[f"dco2_{name}"] = g.co2 - g.co2.shift(k)
            f[f"dtemp_{name}"] = g.temperature - g.temperature.shift(k)
        f["dhum_15"] = g.humidity - g.humidity.shift(3)
        f["co2_floor"] = g.co2.rolling(144, min_periods=36).min()
        f["co2_above_floor"] = g.co2 - f.co2_floor
        f["dco2_slope_change"] = f.dco2_15 - f.dco2_15.shift(3)
        h = (g.index.hour + g.index.minute / 60) / 24 * 2 * np.pi
        f["hsin"], f["hcos"] = np.sin(h), np.cos(h)
        f["work"] = calendar(g.index)[:, 2]
        busy = busy_flags(g.index, cfg["timetable"])
        soon = busy_flags(g.index + pd.Timedelta("30min"), cfg["timetable"])
        f["sched_busy"], f["sched_soon"] = busy, np.maximum(busy, soon)
        return f.iloc[-1]

    def state(self, g, cfg):
        import state as S
        r = self.classifier_feats(g, cfg)
        res = {}
        if self.m.occ is not None and not r[S.OCC_X].isna().any():
            oh = [1.0 if s == cfg["model_space"] else 0.0 for s in self.m.cls_spaces]
            p = float(self.m.occ.predict_proba([list(r[S.OCC_X].values) + oh])[0, 1])
            res["occupancy"] = dict(occupied=p >= 0.5, prob=round(p, 3), method="분류기")
        else:
            res["occupancy"] = dict(occupied=bool(r.dco2_15 >= 30), prob=None, method="규칙: 15분 CO2 +30ppm")
        if self.m.vent is not None and not r[S.VENT_X].isna().any():
            oh = [1.0 if s == cfg["model_space"] else 0.0 for s in self.m.cls_spaces]
            p = float(self.m.vent.predict_proba([list(r[S.VENT_X].values) + oh])[0, 1])
            res["vent_alarm"] = dict(alarm=bool(r.co2 < LIMIT and p >= self.m.vent_thr), prob=round(p, 3), method="분류기")
        else:
            res["vent_alarm"] = dict(alarm=bool(r.co2 >= 850 and r.co2 < LIMIT), prob=None, method="규칙: CO2 ≥ 850")
        res["vent_alarm"]["already_over"] = bool(r.co2 >= LIMIT)
        return res

    def compute_space(self, con, space, now):
        cfg = SPACES[space]
        raw = read_raw(con, cfg["device"])
        period = float(raw.index.to_series().diff().dt.total_seconds().median()) if len(raw) > 3 else 60.0
        alerts, age = rule_alerts(raw, clean(raw) if len(raw) else raw, now, period)
        out = dict(space=space, alias=cfg["alias"], ts=now.isoformat(), device=cfg["device"],
                   data=dict(last_received=raw.index.max().isoformat() if len(raw) else None, age_s=None if math.isinf(age) else round(age)))
        if any(a["type"] == "데이터끊김" for a in alerts) or len(raw) == 0:
            out.update(status="데이터끊김", anomaly=dict(is_anomaly=True, score=None, cause="*", rules=alerts),
                       forecast=None, alerts=None, occupancy=None, vent_alarm=None)
            return out
        g = grid5(clean(raw), now)
        cur = g.iloc[-1]
        out["data"].update(co2=None if pd.isna(cur.co2) else round(float(cur.co2), 1),
                           temperature=None if pd.isna(cur.temperature) else round(float(cur.temperature), 2),
                           humidity=None if pd.isna(cur.humidity) else round(float(cur.humidity), 1),
                           bin_end=(g.index[-1] + pd.Timedelta("5min")).isoformat())
        if g[["co2", "temperature"]].iloc[-1].isna().any():
            out.update(status="데이터부족", anomaly=dict(is_anomaly=bool(alerts), score=None, cause=None, rules=alerts),
                       forecast=None, alerts=None, occupancy=None, vent_alarm=None)
            return out
        fc = self.forecast(g, cfg)
        steps = [5 * (i + 1) for i in range(H)]
        eta = next((s for s, v in zip(steps, fc["co2"]) if v >= LIMIT), None)
        out["forecast"] = dict(step_min=5, horizon_min=steps, base_time=out["data"]["bin_end"], **fc)
        out["alerts"] = dict(co2_over_1000_in_min=eta, message=(f"약 {eta}분 뒤 CO2 1,000ppm 초과 예상" if eta and cur.co2 < LIMIT else
                                                               ("지금 CO2 1,000ppm 이상" if cur.co2 >= LIMIT else None)))
        # 이상 점수: 30분 전에 내놓은 '지금 칸' 예측과 실제의 차이
        t_now = g.index[-1]
        prev = self.past_fc[space].get(t_now)
        score, cause = None, None
        if prev is not None:
            r = {"co2": abs(cur.co2 - prev[0]) / TYPICAL_ERR["co2"], "temperature": abs(cur.temperature - prev[1]) / TYPICAL_ERR["temperature"]}
            cause = max(r, key=r.get)
            score = round(float(r[cause]), 2)
        base = g.index[-1]
        if len(fc["co2"]) >= 6:
            self.past_fc[space][base + pd.Timedelta("30min")] = (fc["co2"][5], fc["temperature"][5])
        self.past_fc[space] = {k: v for k, v in self.past_fc[space].items() if k >= base - pd.Timedelta("1h")}
        rule_cause = alerts[0]["sensor"] if alerts else None
        out["anomaly"] = dict(is_anomaly=bool(alerts) or (score is not None and score > ANOM_THR), score=score,
                              cause=rule_cause or (cause if score is not None and score > ANOM_THR else None),
                              rules=alerts, method="규칙 + 30분 전 예측과의 차이(예측 오차 탐지기)")
        out.update(self.state(g, cfg))
        out["self_check"] = self.self_check(space, g, fc)
        out["status"] = "정상"
        return out

    def cycle(self):
        now = pd.Timestamp.now(tz=TZ)
        t0 = time.perf_counter()
        results, errors = {}, {}
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=10)
        try:
            for sp in SPACES:
                try:
                    results[sp] = self.compute_space(con, sp, now)
                except Exception as e:                       # 한 공간이 실패해도 나머지는 계속
                    errors[sp] = f"{type(e).__name__}: {e}"
                    log.error("계산 실패 %s: %s", sp, traceback.format_exc(limit=3))
                    results[sp] = dict(space=sp, ts=now.isoformat(), status="계산실패", error=errors[sp])
        finally:
            con.close()
        dur = (time.perf_counter() - t0) * 1000
        self.latest = results
        self.health = dict(ts=now.isoformat(), cycle_ms=round(dur, 1), errors=errors, notes=self.m.notes)
        return results, dur, errors


# ------------------------------------------------------------------ 내보내기: MQTT
class Publisher:
    def __init__(self, cfg):
        self.ok = False
        try:
            import paho.mqtt.client as mqtt
            self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"team2-service-{os.getpid()}")
            if cfg.get("username"):
                self.c.username_pw_set(cfg["username"], cfg.get("password"))
            self.c.reconnect_delay_set(1, 30)
            self.c.on_connect = lambda *a: setattr(self, "ok", True)
            self.c.on_disconnect = lambda *a: setattr(self, "ok", False)
            self.c.connect_async(cfg["host"], int(cfg.get("port", 1883)), keepalive=30)
            self.c.loop_start()
        except Exception as e:
            log.error("MQTT 준비 실패: %s", e)
            self.c = None

    def send(self, topic, payload, retain=True):
        if self.c is None:
            return False
        r = self.c.publish(topic, json.dumps(payload, ensure_ascii=False, default=str), qos=1, retain=retain)
        return r.rc == 0


def alert_events(svc, sp, res, now):
    """경보는 새로 생기거나 사라질 때 보내고, 계속 울리는 중이면 30분마다 한 번 다시 보낸다."""
    active = {}
    for a in (res.get("anomaly") or {}).get("rules", []) or []:
        active[f"{a['type']}:{a['sensor']}"] = a
    if (res.get("anomaly") or {}).get("is_anomaly") and (res.get("anomaly") or {}).get("score"):
        active["이상점수:" + str(res["anomaly"]["cause"])] = dict(type="이상점수", sensor=res["anomaly"]["cause"], detail=res["anomaly"]["score"])
    if (res.get("self_check") or {}).get("degraded"):
        active["예측성능저하"] = dict(type="예측성능저하", sensor="co2", detail="; ".join(res["self_check"]["reasons"]))
    if (res.get("vent_alarm") or {}).get("alarm"):
        active["환기선제경보"] = dict(type="환기선제경보", sensor="co2", detail=(res.get("alerts") or {}).get("message"))
    ev, st = [], svc.alert_state[sp]
    for k, a in active.items():
        if k not in st:
            ev.append(dict(event="시작", **a)); st[k] = now
        elif now - st[k] >= pd.Timedelta("30min"):
            ev.append(dict(event="계속", **a)); st[k] = now
    for k in list(st):
        if k not in active:
            ev.append(dict(event="해제", type=k.split(":")[0], sensor=k.split(":")[-1])); del st[k]
    return ev


# ------------------------------------------------------------------ 내보내기: 조회 API
def make_api(svc):
    by_alias = {v["alias"]: k for k, v in SPACES.items()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False, default=str, indent=1).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
            q = parse_qs(u.query)
            if not parts:
                return self.send_json(dict(service="team2 예측 서비스 v1", endpoints=["/spaces", "/state/<공간>", "/forecast/<공간>?minutes=30", "/selfcheck", "/health"],
                                           spaces={k: v["alias"] for k, v in SPACES.items()}))
            if parts[0] == "health":
                return self.send_json(getattr(svc, "health", {}))
            if parts[0] == "selfcheck":
                return self.send_json({k: (svc.latest.get(k, {}).get("self_check") or svc.sc_summary(k)) for k in SPACES})
            if parts[0] == "spaces":
                return self.send_json({k: dict(alias=v["alias"], status=svc.latest.get(k, {}).get("status")) for k, v in SPACES.items()})
            if parts[0] in ("state", "forecast") and len(parts) == 2:
                sp = by_alias.get(parts[1], parts[1])
                if sp not in svc.latest:
                    return self.send_json(dict(error=f"알 수 없는 공간: {parts[1]}", spaces=list(SPACES)), 404)
                res = svc.latest[sp]
                if parts[0] == "state":
                    return self.send_json(res)
                fc = res.get("forecast")
                if not fc:
                    return self.send_json(dict(space=sp, error="지금은 예측 없음", status=res.get("status")), 503)
                mins = int(q.get("minutes", ["60"])[0])
                k = min(range(H), key=lambda i: abs(fc["horizon_min"][i] - mins))
                return self.send_json(dict(space=sp, minutes=fc["horizon_min"][k], base_time=fc["base_time"], co2=fc["co2"][k],
                                           temperature=fc["temperature"][k], unit=dict(co2="ppm", temperature="°C"), method=fc["method"]))
            return self.send_json(dict(error="없는 주소"), 404)
    return Handler


# ------------------------------------------------------------------ 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(COLLECTOR / "data" / "collector.db"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    (BASE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(BASE / "logs" / "service.log", encoding="utf-8")])
    sys.path.insert(0, str(BASE))
    models = Models()
    for n in models.notes:
        log.info(n)
    svc = Service(a.db, models)
    if a.once:
        res, dur, err = svc.cycle()
        print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        print(f"처리 시간 {dur:.0f} ms, 오류 {err}")
        return

    mqtt_cfg = {}
    try:
        sys.path.insert(0, str(COLLECTOR))
        from collector import load_config
        mqtt_cfg = load_config(COLLECTOR / "config.yaml")["mqtt"]
    except Exception as e:
        log.error("수집기 MQTT 설정 읽기 실패: %s", e)
    pub = Publisher(mqtt_cfg) if mqtt_cfg else None
    api = ThreadingHTTPServer(("0.0.0.0", a.port), make_api(svc))
    threading.Thread(target=api.serve_forever, daemon=True).start()
    log.info("조회 API 시작: http://0.0.0.0:%d", a.port)

    cyc = open(BASE / "logs" / "cycles.jsonl", "a", encoding="utf-8")
    while True:
        try:
            res, dur, err = svc.cycle()
            now = pd.Timestamp.now(tz=TZ)
            sent, events = 0, []
            for sp, r in res.items():
                if pub and pub.send(f"{TOPIC}/{sp}", r):
                    sent += 1
                for ev in alert_events(svc, sp, r, now):
                    events.append(dict(space=sp, **ev))
                    if pub:
                        pub.send(f"{TOPIC}/{sp}/alert", dict(space=sp, ts=now.isoformat(), **ev), retain=False)
            if pub:
                pub.send(f"{TOPIC}/_health", svc.health)
            cyc.write(json.dumps(dict(ts=now.isoformat(), ms=round(dur, 1), status={k: v.get("status") for k, v in res.items()},
                                      errors=err, mqtt_sent=sent, mqtt_ok=bool(pub and pub.ok), events=events), ensure_ascii=False, default=str) + "\n")
            cyc.flush()
            if events:
                log.info("경보 %s", events)
        except Exception:
            log.error("주기 실패(다음 주기에 다시): %s", traceback.format_exc(limit=3))
        time.sleep(60 - time.time() % 60 + 1)       # 매분 1초


if __name__ == "__main__":
    main()
