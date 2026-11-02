#!/usr/bin/env python3
"""
과제 7 (분류 모델) — 과제 5 분류기 2개(사용 여부, 환기 경보)를 변환해도
판단이 그대로인지 확인한다. 확인 기준: 분류 결과가 바뀐 비율 0.1% 이하.

분류기는 결정 트리 묶음(HistGradientBoosting)이라 TensorRT 가 바로 못 읽는다.
  - skl2onnx    : 트리를 ONNX 트리 연산으로 → ONNX Runtime (CPU)
  - hummingbird : 트리를 행렬 계산으로 바꿈 → PyTorch GPU, ONNX → TensorRT FP16

설치 (젯슨, 한 번만)
  ~/collector/venv/bin/pip install skl2onnx hummingbird-ml

실행
  ~/collector/venv/bin/python edge_cls.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import state as S

WARM, MIN_N, MIN_SEC = 20, 200, 2.0


def bench(fn):
    for _ in range(WARM):
        fn()
    t, start = [], time.perf_counter()
    while len(t) < MIN_N or time.perf_counter() - start < MIN_SEC:
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return round(float(np.mean(t)), 3), round(float(np.percentile(t, 95)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-trt", action="store_true", help="TensorRT 변환을 건너뜀")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier

    ns = argparse.Namespace(csv="synthetic.csv", schedule="schedule.csv", faults="faults.csv", truth="occupancy_truth.csv")
    d, sch, flt, occ = S.load(ns)
    spaces = sorted(d.space.unique())
    allf = pd.concat([S.frame(d, sch, flt, occ, s) for s in spaces])
    tr, te = S.split(allf, S.TRAIN), S.split(allf, S.TEST)

    # 서비스(과제 8)와 같은 방식으로 학습
    Xo_tr = np.hstack([tr[S.OCC_X].values, S.onehot(tr, spaces)]).astype(np.float32)
    Xo_te = np.hstack([te[S.OCC_X].values, S.onehot(te, spaces)]).astype(np.float32)
    occ_m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xo_tr, tr.label_sched)
    v, vt = tr[tr.vent_valid], te[te.vent_valid]
    w = np.where(v.vent_pos == 1, (1 - v.vent_pos.mean()) / v.vent_pos.mean(), 1.0)
    Xv_tr = np.hstack([v[S.VENT_X].values, S.onehot(v, spaces)]).astype(np.float32)
    Xv_te = np.hstack([vt[S.VENT_X].values, S.onehot(vt, spaces)]).astype(np.float32)
    vent_m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xv_tr, v.vent_pos, sample_weight=w)
    models = {"사용 여부": (occ_m, Xo_te), "환기 경보": (vent_m, Xv_te)}
    print(f"시험 주: 사용 여부 {len(Xo_te):,}칸, 환기 경보 {len(Xv_te):,}칸 | 판단 기준: 확률 ≥ 0.5")

    rows = []

    def record(task, way, proba_fn, X, ref_label, ref_p, note=""):
        p = proba_fn(X)
        lab = (p >= 0.5).astype(int)
        changed = float(np.mean(lab != ref_label))
        m1, p1 = bench(lambda: proba_fn(X[:1]))
        m4, p4 = bench(lambda: proba_fn(X[:4]))
        rows.append(dict(모델=task, 방식=way, 바뀐비율_퍼=round(changed * 100, 3), 바뀐칸=int(np.sum(lab != ref_label)),
                         확률최대차이=round(float(np.max(np.abs(p - ref_p))), 5),
                         공간1_평균ms=m1, 공간1_느린5퍼ms=p1, 공간4_평균ms=m4, 공간4_느린5퍼ms=p4, 비고=note))
        print(f"  {task} | {way}: 바뀐 비율 {changed*100:.3f}% | 공간1 {m1}ms 공간4 {m4}ms")

    for task, (m, X) in models.items():
        ref_p = m.predict_proba(X)[:, 1]
        ref_label = (ref_p >= 0.5).astype(int)
        record(task, "1. scikit-learn CPU (기준)", lambda x: m.predict_proba(x)[:, 1], X, ref_label, ref_p)

        # ---- hummingbird: 트리 → 행렬 계산 (PyTorch GPU, TensorRT)
        try:
            import torch
            from hummingbird.ml import convert
            hb = convert(m, "torch", X[:4])
            hb.to("cuda")
            record(task, "2. PyTorch GPU (hummingbird)", lambda x: hb.predict_proba(x)[:, 1], X, ref_label, ref_p)
        except Exception as e:
            print(f"  {task} PyTorch GPU 실패: {type(e).__name__}: {str(e)[:200]}")
            rows.append(dict(모델=task, 방식="2. PyTorch GPU (hummingbird)", 비고=f"실패: {type(e).__name__}"))

        # ---- ONNX Runtime: skl2onnx 가 안 되면 hummingbird 가 만든 ONNX 를 씀
        tag = 'occ' if task == '사용 여부' else 'vent'
        ox = out / f"cls_{tag}_hb.onnx"
        try:
            import torch
            from hummingbird.ml import convert
            tm = convert(m, "torch", X[:4]).model.eval()
            torch.onnx.export(tm, (torch.from_numpy(X[:4]),), ox, input_names=["x"], dynamic_axes={"x": {0: "b"}}, opset_version=17, dynamo=False)
        except Exception as e:
            print(f"  {task} hummingbird ONNX 내보내기 실패: {type(e).__name__}: {str(e)[:200]}")
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        done = False
        try:
            from skl2onnx import to_onnx
            onx = to_onnx(m, X[:1], options={id(m): {"zipmap": False}})
            path = out / f"cls_{tag}.onnx"
            path.write_bytes(onx.SerializeToString())
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name
            record(task, "3. ONNX Runtime (skl2onnx)", lambda x: sess.run(None, {iname: x})[1][:, 1], X, ref_label, ref_p)
            done = True
        except Exception as e:
            print(f"  {task} skl2onnx 실패 → hummingbird ONNX 로 재시도 ({type(e).__name__})")
        if not done and ox.exists():
            try:
                sess2 = ort.InferenceSession(str(ox), so, providers=["CPUExecutionProvider"])
                outs2 = [o.name for o in sess2.get_outputs()]
                def r2(x):
                    res = sess2.run(None, {"x": x})
                    prob = next(t for t in res if getattr(t, "ndim", 0) == 2 and t.shape[1] == 2)
                    return prob[:, 1]
                record(task, "3. ONNX Runtime (hummingbird)", r2, X, ref_label, ref_p)
            except Exception as e:
                print(f"  {task} ONNX Runtime 실패: {type(e).__name__}: {str(e)[:200]}")
                rows.append(dict(모델=task, 방식="3. ONNX Runtime", 비고=f"실패: {type(e).__name__}"))

        try:
            import torch
            import tensorrt as trt
            if a.no_trt:
                raise RuntimeError("--no-trt 로 건너뜀")
            t0 = time.time()
            log = trt.Logger(trt.Logger.ERROR)
            b = trt.Builder(log); net = b.create_network(0); ps = trt.OnnxParser(net, log)
            if not ps.parse(ox.read_bytes()):
                raise RuntimeError("; ".join(str(ps.get_error(i)) for i in range(ps.num_errors)))
            cfg = b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); cfg.builder_optimization_level = 1
            prof = b.create_optimization_profile(); nf = X.shape[1]
            prof.set_shape("x", (1, nf), (4, nf), (1024, nf)); cfg.add_optimization_profile(prof)
            eng = trt.Runtime(log).deserialize_cuda_engine(b.build_serialized_network(net, cfg))
            print(f"  {task} TensorRT 엔진 생성 {time.time()-t0:.0f}초")
            ctx = eng.create_execution_context()
            outs = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
            stream = torch.cuda.Stream()

            def run(x):
                res = []
                for i in range(0, len(x), 1024):
                    xg = torch.from_numpy(np.ascontiguousarray(x[i:i + 1024])).cuda()
                    ctx.set_input_shape("x", tuple(xg.shape)); ctx.set_tensor_address("x", xg.data_ptr())
                    bufs = {}
                    for o in outs:
                        shp = tuple(ctx.get_tensor_shape(o))
                        dt = torch.float32 if eng.get_tensor_dtype(o) in (trt.float32, trt.float16) else torch.int64
                        bufs[o] = torch.empty(shp, dtype=torch.float32 if eng.get_tensor_dtype(o) == trt.float32 else (torch.float16 if eng.get_tensor_dtype(o) == trt.float16 else torch.int32 if eng.get_tensor_dtype(o) == trt.int32 else dt), device="cuda")
                        ctx.set_tensor_address(o, bufs[o].data_ptr())
                    ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
                    prob = next(t for t in bufs.values() if t.dim() == 2 and t.shape[1] == 2)
                    res.append(prob[:, 1].float().cpu().numpy())
                return np.concatenate(res)
            record(task, "4. TensorRT FP16 (hummingbird)", run, X, ref_label, ref_p)
        except Exception as e:
            print(f"  {task} TensorRT 실패: {type(e).__name__}: {str(e)[:300]}")
            rows.append(dict(모델=task, 방식="4. TensorRT FP16 (hummingbird)", 비고=f"실패: {type(e).__name__}"))

    res = pd.DataFrame(rows)
    res.to_csv(out / "edge_cls_compare.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\n== 과제 7 분류 모델 비교표 (확인 기준: 바뀐 비율 0.1% 이하) ==")
    print(res.drop(columns=["확률최대차이"], errors="ignore").to_string(index=False))
    for _, r in res.dropna(subset=["바뀐비율_퍼"]).iterrows():
        if "기준" not in r.방식:
            print(f"  {r.모델} | {r.방식}: {'통과' if r.바뀐비율_퍼 <= 0.1 else '미달'} ({r.바뀐비율_퍼}%)")


if __name__ == "__main__":
    main()
