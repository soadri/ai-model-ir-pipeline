"""
커스텀 모델 Level 3 NPU 검증 유틸리티

기존 파이프라인 코드를 수정하지 않고,
커스텀 모델의 Linear 레이어를 기존 gemm 커널로 검증.

사용:
  from rngd_pipeline.custom_npu_verify import verify_custom_linears
  results = verify_custom_linears(model, x)
"""
import torch
import struct
import re
import time

from .kernel_backend import run_kernel

GEMM_M, GEMM_K, GEMM_N = 512, 64, 512


def f32_to_bf16_bytes(t):
    """f32 텐서 → bf16 바이너리"""
    flat = t.flatten().tolist()
    r = bytearray()
    for v in flat:
        r.extend(struct.pack('<f', v)[2:4])
    return bytes(r)


def truncate_to_bf16(t):
    """bf16 정밀도로 truncate"""
    return t.to(torch.bfloat16).to(torch.float32)


def _check_compatible(weight_shape):
    """커널과 호환되는 shape인지 판별
    
    Returns:
        (compatible, reason, strategy)
        strategy: 'exact', 'pad', 'tile', None
    """
    out_features, in_features = weight_shape
    
    # Phase 1: 정확히 일치
    if in_features == GEMM_K and out_features == GEMM_N:
        return True, "shape 정확 일치", "exact"
    
    # Phase 1: K가 64 이하이고 N이 512 이하 → 패딩으로 해결
    if in_features <= GEMM_K and out_features <= GEMM_N:
        return True, f"패딩 필요 (K:{in_features}→{GEMM_K}, N:{out_features}→{GEMM_N})", "pad"
    
    # Phase 2: K-tiling (K가 64의 배수, N<=512 패딩)
    if in_features % GEMM_K == 0 and out_features <= GEMM_N:
        k_tiles = in_features // GEMM_K
        return True, f"K-tiling {k_tiles}타일 (K:{in_features}/{GEMM_K}={k_tiles}), N:{out_features} pad→{GEMM_N}", "tile"
    
    # Phase 2: K-tiling + N-tiling (둘 다 배수)
    if in_features % GEMM_K == 0 and out_features % GEMM_N == 0:
        k_tiles = in_features // GEMM_K
        n_tiles = out_features // GEMM_N
        return True, f"2D-tiling K:{k_tiles}xN:{n_tiles}타일", "tile_2d"
    
    # 현재 미지원
    return False, f"미지원 shape (K={in_features}, N={out_features})", None


def _pad_to_kernel_shape(t, target_rows, target_cols):
    """텐서를 커널 shape에 맞게 zero-padding"""
    rows, cols = t.shape
    padded = torch.zeros(target_rows, target_cols)
    padded[:rows, :cols] = t
    return padded


def _run_single_gemm(x_data, w_data, expected_data, label="gemm"):
    """단일 gemm 커널 실행 — 커스텀 데이터를 주입해 실행하고 원본은 자동 복원

    Returns:
        {"passed": bool, "max_diff": float, "elapsed": float, "detail": str}
    """
    inputs = {
        "golden_ffn_gate_a.bin": f32_to_bf16_bytes(x_data),
        "golden_ffn_gate_b.bin": f32_to_bf16_bytes(w_data),
        "golden_ffn_gate_expected.bin": f32_to_bf16_bytes(expected_data),
    }

    t0 = time.time()
    passed, log, _stats = run_kernel("pilot_golden_ffn_gate", inputs=inputs)
    elapsed = time.time() - t0

    # diff 추출
    diffs = re.findall(r"diff=([\d.e+-]+)", log)
    max_diff = max(float(d) for d in diffs) if diffs else None

    # expected/actual 추출
    expected_vals = re.findall(r"expected=bf16\(([-\d.e]+)\)", log)
    actual_vals = re.findall(r"actual=bf16\(([-\d.e]+)\)", log)

    detail = f"{label}: {'PASS' if passed else 'FAIL'}"
    if expected_vals:
        detail += f", sample: expected={expected_vals[0]} actual={actual_vals[0]}"

    return {
        "passed": passed,
        "max_diff": max_diff,
        "elapsed": round(elapsed, 2),
        "detail": detail,
        "log_tail": "\n".join(log.strip().split('\n')[-5:]),
    }


def verify_custom_linears(model, x):
    """커스텀 모델의 Linear 레이어들을 NPU 시뮬레이터로 검증
    
    Args:
        model: PyTorch nn.Module
        x: 입력 텐서
        
    Returns:
        {"results": [...], "summary": {...}}
    """
    results = []
    
    # 모델 실행하여 중간값 캡처
    captures = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                captures[name] = {
                    "input": inp.detach().clone(),
                    "output": out.detach().clone(),
                    "weight": module.weight.data.detach().clone() if hasattr(module, 'weight') else None,
                }
        return hook_fn
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hook_name = name if name else '_root'
            hooks.append(module.register_forward_hook(make_hook(hook_name)))
    
    with torch.no_grad():
        model(x)
    
    for h in hooks:
        h.remove()
    
    # 각 Linear 레이어 검증
    for name, cap in captures.items():
        w = cap["weight"]
        if w is None:
            continue
        
        compatible, reason, strategy = _check_compatible(w.shape)
        
        if not compatible:
            results.append({
                "layer": name,
                "shape": f"Linear({w.shape[1]}, {w.shape[0]})",
                "compatible": False,
                "reason": reason,
                "npu_result": None,
            })
            continue
        
        if strategy == "exact":
            # shape 정확 일치 — 바로 실행
            inp = cap["input"]
            # 입력을 M=512에 맞추기
            if inp.dim() == 2 and inp.shape[0] != GEMM_M:
                inp = _pad_to_kernel_shape(inp, GEMM_M, GEMM_K)
            elif inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
                if inp.shape[0] != GEMM_M:
                    inp = _pad_to_kernel_shape(inp, GEMM_M, GEMM_K)
            
            x_b = truncate_to_bf16(inp)
            w_b = truncate_to_bf16(w)
            exp = truncate_to_bf16(torch.matmul(x_b, w_b.T))
            
            npu_result = _run_single_gemm(inp, w, exp, label=f"{name} Linear({w.shape[1]},{w.shape[0]})")
            
            results.append({
                "layer": name,
                "shape": f"Linear({w.shape[1]}, {w.shape[0]})",
                "compatible": True,
                "reason": reason,
                "strategy": strategy,
                "npu_result": npu_result,
            })
        
        elif strategy == "pad":
            # 패딩 필요
            inp = cap["input"]
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            
            # 입력 패딩: [?, K] → [512, 64]
            inp_pad = _pad_to_kernel_shape(inp[:GEMM_M] if inp.shape[0] >= GEMM_M else inp, GEMM_M, GEMM_K)
            # 가중치 패딩: [N, K] → [512, 64]
            w_pad = _pad_to_kernel_shape(w, GEMM_N, GEMM_K)
            
            x_b = truncate_to_bf16(inp_pad)
            w_b = truncate_to_bf16(w_pad)
            exp = truncate_to_bf16(torch.matmul(x_b, w_b.T))
            
            npu_result = _run_single_gemm(inp_pad, w_pad, exp, label=f"{name} Linear({w.shape[1]},{w.shape[0]}) padded")
            
            results.append({
                "layer": name,
                "shape": f"Linear({w.shape[1]}, {w.shape[0]})",
                "compatible": True,
                "reason": reason,
                "strategy": strategy,
                "npu_result": npu_result,
            })
        
        elif strategy == "tile":
            # K-tiling: K를 64씩 분할, 각 타일 gemm 후 accumulate
            inp = cap["input"]
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            
            K_total = w.shape[1]
            N_total = w.shape[0]
            k_tiles = K_total // GEMM_K
            
            # 입력 M=512 맞추기
            if inp.shape[0] < GEMM_M:
                inp_full = torch.zeros(GEMM_M, K_total)
                inp_full[:inp.shape[0], :] = inp[:, :K_total]
                inp = inp_full
            else:
                inp = inp[:GEMM_M, :K_total]
            
            # 가중치 N 패딩
            w_padded = torch.zeros(GEMM_N, K_total)
            w_padded[:N_total, :] = w
            
            # 패딩 효율
            pad_ratio = round(N_total / GEMM_N * 100, 1)
            
            # PyTorch 참조값 (누적 후 기대값)
            inp_bf16 = truncate_to_bf16(inp)
            w_bf16 = truncate_to_bf16(w_padded)
            pytorch_accumulated = truncate_to_bf16(torch.matmul(inp_bf16, w_bf16.T))
            
            # 타일별 실행
            all_passed = True
            tile_metrics = []
            
            for t in range(k_tiles):
                k_start = t * GEMM_K
                k_end = k_start + GEMM_K
                
                x_tile = inp[:, k_start:k_end]
                w_tile = w_padded[:, k_start:k_end]
                partial_expected = truncate_to_bf16(torch.matmul(
                    truncate_to_bf16(x_tile), truncate_to_bf16(w_tile).T
                ))
                
                tile_result = _run_single_gemm(
                    x_tile, w_tile, partial_expected,
                    label=f"{name} tile[{t}] K={k_start}:{k_end}"
                )
                
                tile_metrics.append({
                    "tile": t,
                    "k_range": f"{k_start}:{k_end}",
                    "passed": tile_result["passed"],
                    "elapsed": tile_result.get("elapsed", 0),
                })
                if not tile_result["passed"]:
                    all_passed = False
            
            total_elapsed = round(sum(m["elapsed"] for m in tile_metrics), 2)
            avg_tile_time = round(total_elapsed / k_tiles, 2) if k_tiles > 0 else 0
            
            # 누적 오차 검증: 타일별 partial sum을 합산하여 PyTorch 결과와 비교
            # PyTorch 참조: bf16 truncate 후 전체 matmul
            pytorch_ref = truncate_to_bf16(torch.matmul(inp_bf16, w_bf16.T))
            
            # NPU 타일별 partial을 bf16 정밀도로 합산 (실제 NPU 동작 근사)
            npu_accumulated = torch.zeros(GEMM_M, GEMM_N)
            for t in range(k_tiles):
                k_start = t * GEMM_K
                k_end = k_start + GEMM_K
                x_tile_bf16 = truncate_to_bf16(inp[:, k_start:k_end])
                w_tile_bf16 = truncate_to_bf16(w_padded[:, k_start:k_end])
                partial = truncate_to_bf16(torch.matmul(x_tile_bf16, w_tile_bf16.T))
                npu_accumulated += partial
            npu_accumulated = truncate_to_bf16(npu_accumulated)
            
            # 누적 오차 계산
            accum_diff = torch.abs(pytorch_ref - npu_accumulated)
            accum_max_diff = float(accum_diff.max())
            accum_mean_diff = float(accum_diff.mean())
            ref_max_val = float(pytorch_ref.abs().max())
            accum_rel_err = round(accum_max_diff / max(ref_max_val, 1e-10) * 100, 4)
            
            # 유효 영역만 (패딩 제외)
            valid_diff = torch.abs(pytorch_ref[:, :N_total] - npu_accumulated[:, :N_total])
            valid_max_diff = float(valid_diff.max())
            valid_ref_max = float(pytorch_ref[:, :N_total].abs().max())
            valid_rel_err = round(valid_max_diff / max(valid_ref_max, 1e-10) * 100, 4)
            
            npu_result = {
                "passed": all_passed,
                "elapsed": total_elapsed,
                "detail": f"{name} Linear({K_total},{N_total}): K-tiling {k_tiles}타일 전부 {'PASS' if all_passed else 'FAIL'}",
                "tiling_metrics": {
                    "k_tiles": k_tiles,
                    "n_tiles": 1,
                    "total_tiles": k_tiles,
                    "tile_time_avg": avg_tile_time,
                    "tile_time_total": total_elapsed,
                    "pad_efficiency": pad_ratio,
                    "tile_results": tile_metrics,
                    "accumulation": {
                        "max_diff": round(accum_max_diff, 6),
                        "mean_diff": round(accum_mean_diff, 6),
                        "rel_err_pct": accum_rel_err,
                        "ref_max_val": round(ref_max_val, 2),
                        "valid_max_diff": round(valid_max_diff, 6),
                        "valid_rel_err_pct": valid_rel_err,
                    },
                },
            }
            
            results.append({
                "layer": name,
                "shape": f"Linear({w.shape[1]}, {w.shape[0]})",
                "compatible": True,
                "reason": reason,
                "strategy": strategy,
                "npu_result": npu_result,
            })
        
        else:
            # tile_2d 등 미구현
            results.append({
                "layer": name,
                "shape": f"Linear({w.shape[1]}, {w.shape[0]})",
                "compatible": True,
                "reason": reason,
                "strategy": strategy,
                "npu_result": None,
            })
    
    # 요약
    tested = [r for r in results if r.get("npu_result")]
    passed = [r for r in tested if r["npu_result"]["passed"]]
    skipped = [r for r in results if not r.get("npu_result")]
    
    return {
        "results": results,
        "summary": {
            "total_linears": len(results),
            "tested": len(tested),
            "passed": len(passed),
            "skipped": len(skipped),
            "all_passed": len(passed) == len(tested) and len(tested) > 0,
        }
    }




def _f32_bytes(data):
    """f32 리스트 → 리틀엔디언 바이너리"""
    return struct.pack(f"<{len(data)}f", *data)


def _run_custom_bin(bin_name, inputs, timeout=120):
    """custom_verify bin 실행 — 입력 데이터를 주입해 실행"""
    t0 = time.time()
    passed, log, _stats = run_kernel(bin_name, timeout, inputs=inputs)
    elapsed = time.time() - t0

    # 개별 테스트 결과 파싱
    test_results = {}
    for match in re.finditer(r"PASS: (\w+) max_diff=([\d.e+-]+)", log):
        test_results[match.group(1)] = float(match.group(2))

    return {
        "passed": passed,
        "elapsed": round(elapsed, 2),
        "test_results": test_results,
        "log_tail": "\n".join(log.strip().split("\n")[-5:]),
    }


def verify_custom_rmsnorm(model, x):
    """커스텀 모델에서 RMSNorm/LayerNorm 패턴 검증
    
    A=16 elementwise 커널로 pow2, rsqrt, xnorm 검증
    """
    results = []
    
    # 모델에서 norm 레이어 찾기
    captures = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                captures[name] = {
                    "input": inp.detach().clone(),
                    "output": out.detach().clone(),
                    "type": type(module).__name__,
                }
        return hook_fn
    
    for name, module in model.named_modules():
        mod_type = type(module).__name__.lower()
        if any(k in mod_type for k in ['rmsnorm', 'layernorm', 'norm']):
            if hasattr(module, 'weight'):
                hook_name = name if name else '_root'
                hooks.append(module.register_forward_hook(make_hook(hook_name)))
    
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    
    for name, cap in captures.items():
        inp = cap["input"].flatten()
        
        # A=16 크기로 맞추기
        if inp.numel() < 16:
            padded = torch.zeros(16)
            padded[:inp.numel()] = inp
            inp = padded
        else:
            inp = inp[:16]
        
        inp_list = inp.tolist()
        
        # pow2 = x * x
        pow2_expected = [v * v for v in inp_list]
        
        # mean_sq + rsqrt
        mean_sq = sum(pow2_expected) / len(pow2_expected)
        rsqrt_input = [mean_sq + 1e-6] * 16
        rsqrt_expected = [1.0 / (v ** 0.5) for v in rsqrt_input]
        
        # xnorm = x * rsqrt
        xnorm_expected = [a * b for a, b in zip(inp_list, rsqrt_expected)]
        
        # 커널 입력 데이터 (실행 후 자동 정리)
        inputs = {
            "custom_elem_input.bin": _f32_bytes(inp_list),
            "custom_elem_pow2_expected.bin": _f32_bytes(pow2_expected),
            "custom_elem_rsqrt_input.bin": _f32_bytes(rsqrt_input),
            "custom_elem_rsqrt_expected.bin": _f32_bytes(rsqrt_expected),
            "custom_elem_xnorm_x.bin": _f32_bytes(inp_list),
            "custom_elem_xnorm_rsqrt.bin": _f32_bytes(rsqrt_expected),
            "custom_elem_xnorm_expected.bin": _f32_bytes(xnorm_expected),
        }
        npu_result = _run_custom_bin("custom_verify_elementwise", inputs)
        
        results.append({
            "layer": name,
            "type": cap["type"],
            "shape": f"dim={inp.numel()}",
            "npu_result": npu_result,
        })

    
    tested = len(results)
    passed = sum(1 for r in results if r["npu_result"]["passed"])
    
    return {
        "results": results,
        "summary": {"tested": tested, "passed": passed, "all_passed": tested == passed and tested > 0}
    }


def verify_custom_silu(model, x):
    """커스텀 모델에서 SiLU 활성화 검증
    
    A=8 silu 커널로 검증
    """
    results = []
    
    captures = {}
    hooks = []
    
    # 방법 1: nn.SiLU 모듈 hook
    def make_hook(name):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                captures[name] = {
                    "input": inp.detach().clone(),
                    "output": out.detach().clone(),
                }
        return hook_fn
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.SiLU):
            hook_name = name if name else '_root'
            hooks.append(module.register_forward_hook(make_hook(hook_name)))
    
    # 방법 2: F.silu() 함수 호출 캡처 (monkey-patch)
    import torch.nn.functional as F
    _original_silu = F.silu
    _fsilu_count = [0]
    
    def _patched_silu(input, inplace=False):
        result = _original_silu(input, inplace=inplace)
        name = f"F.silu_{_fsilu_count[0]}"
        _fsilu_count[0] += 1
        if isinstance(input, torch.Tensor):
            captures[name] = {
                "input": input.detach().clone(),
                "output": result.detach().clone(),
            }
        return result
    
    F.silu = _patched_silu
    try:
        with torch.no_grad():
            model(x)
    finally:
        F.silu = _original_silu
    
    for h in hooks:
        h.remove()
    
    for name, cap in captures.items():
        inp = cap["input"].flatten()
        
        # A=8 크기
        if inp.numel() < 8:
            padded = torch.zeros(8)
            padded[:inp.numel()] = inp
            inp = padded
        else:
            inp = inp[:8]
        
        inp_list = inp.tolist()
        silu_expected = [v * (1.0 / (1.0 + (-v).__abs__().__class__((-v).__float__()))) for v in inp_list]
        # 더 간단하게
        import math
        silu_expected = [v / (1.0 + math.exp(-v)) for v in inp_list]
        
        inputs = {
            "custom_silu_input.bin": _f32_bytes(inp_list),
            "custom_silu_expected.bin": _f32_bytes(silu_expected),
        }
        npu_result = _run_custom_bin("custom_verify_silu", inputs)
        
        results.append({
            "layer": name,
            "type": "SiLU",
            "shape": f"A={len(inp_list)}",
            "npu_result": npu_result,
        })

    
    tested = len(results)
    passed = sum(1 for r in results if r["npu_result"]["passed"])
    
    return {
        "results": results,
        "summary": {"tested": tested, "passed": passed, "all_passed": tested == passed and tested > 0}
    }


def verify_custom_model_full(model, x):
    """커스텀 모델 전체 Level 3 검증 — Linear + RMSNorm + SiLU
    
    Returns:
        통합 결과 dict
    """
    all_results = []
    
    # Linear 검증
    linear_result = verify_custom_linears(model, x)
    for r in linear_result["results"]:
        r["op_type"] = "Linear (gemm)"
        all_results.append(r)
    
    # RMSNorm 검증
    norm_result = verify_custom_rmsnorm(model, x)
    for r in norm_result["results"]:
        r["op_type"] = "RMSNorm (elementwise)"
        all_results.append(r)
    
    # SiLU 검증
    silu_result = verify_custom_silu(model, x)
    for r in silu_result["results"]:
        r["op_type"] = "SiLU (elementwise)"
        all_results.append(r)
    
    tested = sum(1 for r in all_results if r.get("npu_result"))
    passed = sum(1 for r in all_results if r.get("npu_result") and r["npu_result"].get("passed"))
    skipped = sum(1 for r in all_results if not r.get("npu_result"))
    
    return {
        "results": all_results,
        "summary": {
            "total_layers": len(all_results),
            "tested": tested,
            "passed": passed,
            "skipped": skipped,
            "all_passed": tested == passed and tested > 0,
        }
    }


if __name__ == "__main__":
    import math
    print("=== 커스텀 모델 Level 3 전체 검증 테스트 ===\n")
    
    torch.manual_seed(7777)
    
    class TestFFN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(64)
            self.gate = torch.nn.Linear(64, 512, bias=False)
            self.silu = torch.nn.SiLU()
            self.up = torch.nn.Linear(64, 512, bias=False)
            self.down = torch.nn.Linear(512, 64, bias=False)
        def forward(self, x):
            h = self.norm(x)
            return self.down(self.silu(self.gate(h)[:,:,:512]) * self.up(h)[:,:,:512])
    
    model = TestFFN()
    model.eval()
    x = torch.randn(1, 8, 64)
    
    result = verify_custom_model_full(model, x)
    
    print(f"요약: {result['summary']}\n")
    for r in result['results']:
        npu = r.get('npu_result', {})
        status = "✅ PASS" if npu and npu.get('passed') else "⏭ SKIP" if not npu else "❌ FAIL"
        print(f"  [{r.get('op_type','?')}] {r['layer']}: {r.get('shape','')} — {status}")
        if npu:
            print(f"    elapsed: {npu.get('elapsed')}s")
