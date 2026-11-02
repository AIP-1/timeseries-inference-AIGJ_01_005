#!/usr/bin/env python3
"""
과제 7 — 엣지 최적화: 과제 3 GRU(60분 예측, 서비스의 이상 점수도 이 예측으로 계산)를
PyTorch CPU / PyTorch GPU / ONNX Runtime / TensorRT FP16 네 가지로 돌려 비교한다.

준비 (젯슨, 한 번만)
  ~/collector/venv/bin/pip install onnx onnxruntime onnxscript
  sudo apt install -y tensorrt python3-libnvinfer
  # venv 가 시스템의 tensorrt 파이썬 부품을 쓰도록 연결
  ln -sf "$(/usr/bin/python3 -c 'import tensorrt,os;print(os.path.dirname(tensorrt.__file__))')" \
         ~/collector/venv/lib/python3.12/site-packages/

실행
  ~/collector/venv/bin/python edge.py            # 네 가지 방식 비교 (+ 심화 INT8)
  sudo jetson_clocks; ~/collector/venv/bin/python edge.py --clock 고정   # 클럭 고정 후 (선택)

각 방식 × 배치(공간 1곳 / 4곳):
  지연(평균·느린 5%) — 입력을 넘기고 결과를 받을 때까지 전부 포함
  기준(PyTorch CPU FP32)과의 최대 차이(ppm), 시험 주 60분 뒤 MAE
  측정 중 전력(VDD_IN)·GPU 사용률·메모리 (tegrastats 0.1초 간격)
"""
import argparse
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

import forecast as F

PROC = psutil.Process()
rss = lambda: PROC.memory_info().rss / 2**20
MIN_N, MIN_SEC, WARM = 200, 3.0, 20


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600).stdout.strip()
    except Exception as e:
        return f"(실패: {e})"


class Tegra:
    """측정하는 동안 tegrastats 를 0.1초마다 읽어 전력·GPU·메모리 평균을 낸다."""
    exe = shutil.which("tegrastats") or ("/usr/bin/tegrastats" if os.path.exists("/usr/bin/tegrastats") else None)

    def __enter__(self):
        self.lines, self.p = [], None
        if self.exe:
            self.p = subprocess.Popen([self.exe, "--interval", "100"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.t = threading.Thread(target=lambda: [self.lines.append(l) for l in self.p.stdout], daemon=True)
            self.t.start()
            time.sleep(0.3)
        return self

    def __exit__(self, *a):
        if self.p:
            self.p.terminate()
            try:
                self.p.wait(2)
            except Exception:
                self.p.kill()

    def summary(self):
        ram = [int(m.group(1)) for l in self.lines if (m := re.search(r"RAM (\d+)/\d+MB", l))]
        gpu = [int(m.group(1)) for l in self.lines if (m := re.search(r"GR3D_FREQ (\d+)%", l))]
        pw = [int(m.group(1)) for l in self.lines if (m := re.search(r"VDD_IN (\d+)mW", l))]
        avg = lambda v: round(float(np.mean(v)), 0) if v else None
        return dict(전력_mW=avg(pw), GPU사용률_퍼=avg(gpu), GPU최대_퍼=max(gpu) if gpu else None, 보드메모리_MB=avg(ram))


def bench(fn):
    for _ in range(WARM):
        fn()
    t, start = [], time.perf_counter()
    while len(t) < MIN_N or time.perf_counter() - start < MIN_SEC:
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return float(np.mean(t)), float(np.percentile(t, 95)), len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--clock", default="고정 안 함", help="클럭 고정 여부 기록용 (sudo jetson_clocks 를 했으면 '고정')")
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(a.threads)

    power = sh("nvpmodel -q 2>/dev/null | head -1") or "(확인 불가)"
    print(f"측정 조건: 전력 모드 {power} | 클럭 {a.clock} | CPU 스레드 {a.threads} | 워밍업 {WARM}회, 최소 {MIN_N}회·{MIN_SEC:.0f}초 반복")

    # ---- 데이터: 과제 3 시험 주 전체
    d, sch, flt = F.load("synthetic.csv", "schedule.csv", "faults.csv")
    grid, idx, spaces = F.clean_grid(d, flt)
    data = F.Data(grid, idx, spaces, sch)
    F.DSD = data.dsd
    te = data.windows("test", 24)
    truth = te["Y0"][:, None, :] + te["Y"] * F.DSD[None, None, :]
    X, Fu = np.ascontiguousarray(te["X"], np.float32), np.ascontiguousarray(te["F"], np.float32)
    batches = {1: (X[:1], Fu[:1]), 4: (X[:4], Fu[:4])}
    NIN, NF = X.shape[2], Fu.shape[2]
    print(f"시험 창 {len(X):,}개")

    to_ppm = lambda delta: delta[..., 0] * F.DSD[0]
    mae60 = lambda delta: round(float(np.mean(np.abs(te["Y0"][:, 0] + delta[:, 11, 0] * F.DSD[0] - truth[:, 11, 0]))), 2)

    with Tegra() as tg:
        time.sleep(3)
    idle = tg.summary()
    print(f"가만히 있을 때: {idle}")

    rows, ref = [], None

    def measure(name, run_batch, run_all, extra=None):
        """run_batch(x, f) -> numpy, run_all() -> 시험 주 전체 delta"""
        nonlocal ref
        delta = run_all()
        if ref is None:
            ref = delta
        diff = float(np.max(np.abs(to_ppm(delta) - to_ppm(ref))))
        for b, (xb, fb) in batches.items():
            with Tegra() as tg:
                m, p, n = bench(lambda: run_batch(xb, fb))
            s = tg.summary()
            rows.append(dict(방식=name, 배치=b, 평균ms=round(m, 3), 느린5퍼ms=round(p, 3), 반복=n,
                             최대차이_ppm=round(diff, 3), CO2_60분_MAE=mae60(delta),
                             전력_mW=s["전력_mW"], 전력증가_mW=None if s["전력_mW"] is None or idle["전력_mW"] is None else s["전력_mW"] - idle["전력_mW"],
                             GPU사용률_퍼=s["GPU사용률_퍼"], 보드메모리_MB=s["보드메모리_MB"], **(extra or {})))
            print(f"  {name} 배치{b}: 평균 {m:.3f}ms, 느린5% {p:.3f}ms | 최대차이 {diff:.3f}ppm | 전력 {s['전력_mW']}mW GPU {s['GPU사용률_퍼']}%")

    # ---- 1. PyTorch CPU
    model = F.build("gru", NIN, NF)
    model.load_state_dict(torch.load("results/gru_main_L24.pt", map_location="cpu")); model.eval()

    def pt_runner(m, dev):
        def r(x, f):
            with torch.inference_mode():
                y = m(torch.from_numpy(x).to(dev), torch.from_numpy(f).to(dev)).cpu().numpy()
            return y
        return r
    r = pt_runner(model, "cpu")
    measure("1. PyTorch CPU", r, lambda: r(X, Fu))

    # ---- 2. PyTorch GPU
    try:
        gm = F.build("gru", NIN, NF)
        gm.load_state_dict(model.state_dict()); gm.eval().to("cuda")
        r = pt_runner(gm, "cuda")
        measure("2. PyTorch GPU", r, lambda: np.concatenate([r(X[i:i + 1024], Fu[i:i + 1024]) for i in range(0, len(X), 1024)]))
    except Exception as e:
        print(f"  2. PyTorch GPU 실패: {type(e).__name__}: {str(e)[:200]}")
        rows.append(dict(방식="2. PyTorch GPU", 배치="-", 비고=f"실패: {type(e).__name__}"))

    # ---- ONNX 내보내기 (3, 4번이 같이 씀)
    onnx_path = out / "gru.onnx"
    x4, f4 = torch.from_numpy(batches[4][0]), torch.from_numpy(batches[4][1])
    try:
        torch.onnx.export(model, (x4, f4), onnx_path, input_names=["x", "f"], output_names=["y"],
                          dynamic_axes={"x": {0: "batch"}, "f": {0: "batch"}, "y": {0: "batch"}}, opset_version=17, dynamo=False)
    except Exception as e:
        print(f"  (기존 방식 ONNX 내보내기 실패 → 새 방식: {type(e).__name__})")
        B = torch.export.Dim("batch")
        torch.onnx.export(model, (x4, f4), onnx_path, input_names=["x", "f"], output_names=["y"],
                          dynamic_shapes={"x": {0: B}, "f": {0: B}}, dynamo=True)

    # ---- 3. ONNX Runtime (CPU)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    r = lambda x, f: sess.run(None, {"x": x, "f": f})[0]
    measure("3. ONNX Runtime", r, lambda: r(X, Fu))

    # ---- 4. TensorRT FP16 (+ 참고용 FP32)
    try:
        import tensorrt as trt
        log = trt.Logger(trt.Logger.WARNING)
        MAXB = 1024

        def build_engine(fp16):
            builder = trt.Builder(log)
            net = builder.create_network(0)
            parser = trt.OnnxParser(net, log)
            if not parser.parse(onnx_path.read_bytes()):
                raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
            if fp16:
                cfg.set_flag(trt.BuilderFlag.FP16)
            prof = builder.create_optimization_profile()
            prof.set_shape("x", (1, 24, NIN), (4, 24, NIN), (MAXB, 24, NIN))
            prof.set_shape("f", (1, F.H, NF), (4, F.H, NF), (MAXB, F.H, NF))
            cfg.add_optimization_profile(prof)
            t0 = time.time()
            blob = builder.build_serialized_network(net, cfg)
            if blob is None:
                raise RuntimeError("엔진 생성 실패")
            path = out / f"gru_{'fp16' if fp16 else 'fp32'}.engine"
            path.write_bytes(bytes(blob))
            eng = trt.Runtime(log).deserialize_cuda_engine(blob)
            return eng, time.time() - t0, path

        def trt_runner(eng):
            ctx = eng.create_execution_context()
            stream = torch.cuda.Stream()

            def r(x, f):
                xg = torch.from_numpy(x).cuda(non_blocking=False)
                fg = torch.from_numpy(f).cuda(non_blocking=False)
                y = torch.empty((len(x), F.H, 2), dtype=torch.float32, device="cuda")
                ctx.set_input_shape("x", tuple(xg.shape)); ctx.set_input_shape("f", tuple(fg.shape))
                ctx.set_tensor_address("x", xg.data_ptr()); ctx.set_tensor_address("f", fg.data_ptr())
                ctx.set_tensor_address("y", y.data_ptr())
                ctx.execute_async_v3(stream.cuda_stream)
                stream.synchronize()
                return y.cpu().numpy()
            return r

        for fp16 in (True, False):
            name = "4. TensorRT FP16" if fp16 else "(참고) TensorRT FP32"
            eng, sec, path = build_engine(fp16)
            print(f"  {name} 엔진 생성 {sec:.0f}초, {path.stat().st_size/1024:.0f}KB")
            r = trt_runner(eng)
            measure(name, r, lambda: np.concatenate([r(X[i:i + MAXB], Fu[i:i + MAXB]) for i in range(0, len(X), MAXB)]),
                    dict(엔진_KB=round(path.stat().st_size / 1024)))
    except Exception as e:
        print(f"  4. TensorRT 실패: {type(e).__name__}: {str(e)[:300]}")
        rows.append(dict(방식="4. TensorRT FP16", 배치="-", 비고=f"실패: {type(e).__name__}"))

    # ---- 심화(B): INT8 (ONNX Runtime 동적 양자화)
    if not a.no_int8:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            q = out / "gru_int8.onnx"
            quantize_dynamic(str(onnx_path), str(q), weight_type=QuantType.QInt8)
            sq = ort.InferenceSession(str(q), so, providers=["CPUExecutionProvider"])
            r = lambda x, f: sq.run(None, {"x": x, "f": f})[0]
            measure("(심화) ONNX Runtime INT8", r, lambda: r(X, Fu))
        except Exception as e:
            print(f"  INT8 건너뜀: {e}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "edge_compare.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(f"\n== 과제 7 비교표 ==  (전력 모드 {power} | 클럭 {a.clock} | 가만히 있을 때 전력 {idle['전력_mW']}mW, 메모리 {idle['보드메모리_MB']}MB)")
    show = [c for c in ["방식", "배치", "평균ms", "느린5퍼ms", "최대차이_ppm", "CO2_60분_MAE", "전력증가_mW", "GPU사용률_퍼", "보드메모리_MB", "비고"] if c in res]
    print(res[show].to_string(index=False))
    ok = res[res["방식"] == "4. TensorRT FP16"]
    if len(ok) and "최대차이_ppm" in ok and ok["최대차이_ppm"].notna().any():
        v = float(ok["최대차이_ppm"].dropna().iloc[0])
        print(f"\n확인 기준: TensorRT FP16 CO2 최대 차이 {v:.3f}ppm → {'통과 (1ppm 이하)' if v <= 1 else '미달 (1ppm 초과)'}")


if __name__ == "__main__":
    main()
