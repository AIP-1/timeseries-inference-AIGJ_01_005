#!/usr/bin/env python3
"""
과제 7 심화 ② — 모델 4개를 동시에 올렸을 때 총 메모리, 로컬 LLM과 함께 올릴 조합

  ~/collector/venv/bin/python mem4.py

모델 4개 (서비스가 쓰는 것)
  예측 GRU            results/gru.onnx  (TensorRT 쪽은 results/gru_fp32.engine)
  사용 여부 분류기     results/cls_occ_hb.onnx
  환기 경보 분류기     results/cls_vent_hb.onnx
  이상 탐지 오토인코더  results/ae.onnx  (없으면 synthetic.csv 로 학습해 만든다, 1~2분)

방식마다 새 프로세스를 띄워 잰다 (앞 방식의 메모리가 섞이지 않게).
  RSS      = 그 프로세스가 차지한 CPU 쪽 메모리
  시스템 Δ  = /proc/meminfo 의 MemAvailable 감소량. 젯슨은 CPU·GPU가 메모리를 같이 쓰므로
             GPU 메모리(CUDA·TensorRT)까지 여기에 잡힌다 → 이 값을 '실제로 줄어든 메모리'로 본다.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
R = BASE / "results"
ONNX = {"예측 GRU": R / "gru.onnx", "사용 여부 분류기": R / "cls_occ_hb.onnx",
        "환기 경보 분류기": R / "cls_vent_hb.onnx", "이상 탐지 AE": R / "ae.onnx"}
ENGINE = R / "gru_fp32.engine"


def avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024
    return float("nan")


def total_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) / 1024


def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def make_ae():
    """과제 4의 오토인코더(MLP)를 합성 데이터 학습 구간으로 학습해 ONNX로 저장."""
    import anomaly as A
    from onnx import TensorProto, helper, numpy_helper
    print("  ae.onnx 없음 → 과제 4 오토인코더를 synthetic.csv 로 학습 (1~2분)", flush=True)
    d, _ = A.load(str(BASE / "synthetic.csv"), str(BASE / "faults.csv"))
    spaces = sorted(d.space.unique())
    ae = A.AE(d, spaces).fit()
    m = ae.model
    n_in = m.coefs_[0].shape[0]
    # MLP = (곱하기+더하기 → ReLU) 반복, 마지막 층은 그대로. ONNX 의 Gemm·Relu 로 그대로 옮긴다.
    nodes, inits, prev = [], [], "x"
    for i, (W_, b_) in enumerate(zip(m.coefs_, m.intercepts_)):
        inits += [numpy_helper.from_array(W_.astype(np.float32), f"W{i}"),
                  numpy_helper.from_array(b_.astype(np.float32), f"b{i}")]
        out = "y" if i == len(m.coefs_) - 1 else f"h{i}"
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"b{i}"], [out + ("" if out == "y" else "_lin")]))
        if out != "y":
            nodes.append(helper.make_node("Relu", [out + "_lin"], [out]))
        prev = out
    g = helper.make_graph(nodes, "ae", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", n_in])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", m.coefs_[-1].shape[1]])], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    ONNX["이상 탐지 AE"].write_bytes(model.SerializeToString())
    # 변환이 맞는지: sklearn 결과와 비교
    import onnxruntime as ort
    X = np.random.rand(16, n_in).astype(np.float32)
    diff = np.abs(ort.InferenceSession(str(ONNX["이상 탐지 AE"]), providers=["CPUExecutionProvider"]).run(None, {"x": X})[0] - m.predict(X)).max()
    print(f"  변환 확인: sklearn 과 최대 차이 {diff:.2e}", flush=True)
    print(f"  저장: {ONNX['이상 탐지 AE']} (입력 {n_in}개)", flush=True)


def feed(sess, batch=4):
    """세션 입력 모양대로 무작위 입력을 만든다 (배치 차원은 batch)."""
    out = {}
    for i in sess.get_inputs():
        shape = [batch if (not isinstance(s, int) or s <= 0) else s for s in i.shape]
        if i.shape and isinstance(i.shape[0], int) and i.shape[0] > 0:
            shape[0] = i.shape[0]
        dt = np.int64 if "int64" in i.type else np.float32
        out[i.name] = np.random.rand(*shape).astype(dt)
    return out


# ------------------------------------------------------------------ 자식 프로세스: 방식 하나 재기
def child(mode):
    a0, r0 = avail_mb(), rss_mb()
    steps = []

    def mark(name):
        time.sleep(0.3)
        steps.append(dict(단계=name, RSS=round(rss_mb(), 1), 시스템감소=round(a0 - avail_mb(), 1)))

    mark("시작(파이썬+numpy)")
    if mode in ("ort", "trt"):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        mark("onnxruntime 불러옴")
        sessions = {}
        for name, p in ONNX.items():
            if mode == "trt" and name == "예측 GRU":
                continue
            s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
            s.run(None, feed(s))
            sessions[name] = s
            mark(f"+ {name} (ORT CPU)")
    if mode in ("trt", "torch_cuda"):
        import torch
        torch.zeros(1, device="cuda"); torch.cuda.synchronize()
        mark("+ CUDA 켬 (torch)")
    if mode == "trt":
        import tensorrt as trt
        log = trt.Logger(trt.Logger.WARNING)
        eng = trt.Runtime(log).deserialize_cuda_engine(ENGINE.read_bytes())
        ctx = eng.create_execution_context()
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        bufs = {}
        for n in names:
            shp = list(eng.get_tensor_shape(n)); shp[0] = 4
            if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                ctx.set_input_shape(n, tuple(shp))
        for n in names:
            shp = tuple(ctx.get_tensor_shape(n))
            bufs[n] = torch.rand(shp, device="cuda")
            ctx.set_tensor_address(n, bufs[n].data_ptr())
        st = torch.cuda.Stream()
        ctx.execute_async_v3(st.cuda_stream); st.synchronize()
        mark("+ 예측 GRU (TensorRT FP32, GPU)")
    if mode == "torch_cpu":
        import torch
        import forecast as F
        mark("torch 불러옴")
        sd = torch.load(R / "gru_main_L24.pt", map_location="cpu")
        state = sd.get("state", sd) if isinstance(sd, dict) else sd
        n_in = state["enc.weight_ih_l0"].shape[1]
        n_fut = (state["head.0.weight"].shape[1] - 64) // F.H
        m = F.build("gru", n_in, n_fut); m.load_state_dict(state); m.eval()
        with torch.no_grad():
            m(torch.rand(4, 24, n_in), torch.rand(4, F.H, n_fut))
        mark("+ 예측 GRU (PyTorch CPU)")
    print("JSON" + json.dumps(steps, ensure_ascii=False))


def run_child(mode):
    p = subprocess.run([sys.executable, __file__, "--child", mode], capture_output=True, text=True, timeout=600)
    for line in p.stdout.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    print(f"  [{mode}] 실패:\n{p.stderr[-800:]}")
    return None


def main():
    if "--child" in sys.argv:
        return child(sys.argv[sys.argv.index("--child") + 1])
    import pandas as pd
    import psutil

    print(f"보드 전체 메모리 {total_mb():,.0f}MB, 지금 사용 가능 {avail_mb():,.0f}MB\n")
    if not ONNX["이상 탐지 AE"].exists():
        make_ae()
    for name, p in ONNX.items():
        print(f"  {name:10s} {p.name:20s} {p.stat().st_size/1024:7.0f}KB" if p.exists() else f"  {name} 파일 없음: {p}")
    print(f"  예측 GRU TensorRT  {ENGINE.name:20s} {ENGINE.stat().st_size/1024:7.0f}KB\n" if ENGINE.exists() else "")

    plans = [("ort", "A. 4개 모두 ONNX Runtime CPU (추천안)"),
             ("trt", "B. GRU만 TensorRT GPU + 나머지 3개 ORT CPU"),
             ("torch_cuda", "(참고) PyTorch로 CUDA만 켰을 때"),
             ("torch_cpu", "(참고) 지금 서비스 방식: GRU PyTorch CPU")]
    summary = []
    for mode, title in plans:
        steps = run_child(mode)
        if not steps:
            continue
        print(f"## {title}")
        df = pd.DataFrame(steps)
        df["RSS증가"] = (df.RSS - df.RSS.iloc[0]).round(1)
        print(df.to_string(index=False), "\n")
        summary.append(dict(방식=title, 프로세스_RSS_MB=df.RSS.iloc[-1], 시스템_감소_MB=df.시스템감소.iloc[-1],
                            모델만_RSS증가_MB=round(df.RSS.iloc[-1] - df.RSS.iloc[0], 1)))

    # 지금 돌고 있는 서비스·수집기 실제 메모리
    live = []
    for p in psutil.process_iter(["cmdline", "memory_info"]):
        cmd = " ".join(p.info["cmdline"] or [])
        for key, label in (("service.py", "예측 서비스(가동 중)"), ("collector.py", "수집기(가동 중)")):
            if key in cmd and "mem4" not in cmd:
                live.append(dict(프로세스=label, PID=p.pid, RSS_MB=round(p.info["memory_info"].rss / 2**20, 1)))
    s = pd.DataFrame(summary)
    s.to_csv(R / "mem4_summary.csv", index=False)
    print("## 요약 (배치 4로 한 번씩 추론한 직후)")
    print(s.to_string(index=False))
    if live:
        print("\n## 지금 가동 중인 프로세스")
        print(pd.DataFrame(live).to_string(index=False))

    tot, av = total_mb(), avail_mb()
    a = s[s.방식.str.startswith("A.")]
    need = float(a.시스템_감소_MB.iloc[0]) if len(a) else float("nan")
    print(f"\n## LLM 예산 (대략)")
    print(f"전체 {tot:,.0f}MB 중 지금 사용 가능 {av:,.0f}MB (수집기·서비스 포함 상태)")
    print(f"시계열 모델 4개(A안)를 새로 올리는 데 드는 양 ≈ {need:,.0f}MB")
    print("로컬 LLM 4비트 양자화의 대략적 크기(가중치 + 문맥 2~4k 기준, 실제는 실행기마다 다름):")
    for nm, gb in (("1B급", 1.0), ("3B급", 2.3), ("7~8B급", 5.0)):
        left = av - gb * 1024
        print(f"  {nm:6s} 약 {gb:.1f}GB → 남는 메모리 약 {left:,.0f}MB {'(여유)' if left > 1500 else '(빠듯)' if left > 500 else '(부족)'}")
    print(f"\n결과: {R/'mem4_summary.csv'}")


if __name__ == "__main__":
    main()
