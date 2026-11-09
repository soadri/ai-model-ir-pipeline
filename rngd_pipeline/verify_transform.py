"""
verify_transform.py — IR 변환 정확성 자동 검증
Level 1: 구조적 검증 (op 수, shape 보존, 의존 관계)
Level 2: 의미적 검증 (PyTorch 기준값 대비 shape/dtype 일관성)

사용법:
  result = verify_transform(model, example_input)
  print_verify_report(result)
"""

import torch
import torch_mlir
from torch_mlir.ir import Context, Module, RankedTensorType
from .e2e_pipeline import rewrite_to_rngd, _list_op_names, mark_remaining
from .e2e_pipeline import extract_dataflow, detect_fusion_patterns, insert_optimization_markers
from dataclasses import dataclass, field
from collections import Counter
import time


@dataclass
class VerifyCheck:
    """개별 검증 항목 결과"""
    name: str
    passed: bool
    expected: str
    actual: str
    detail: str = ""


@dataclass
class VerifyResult:
    """전체 검증 결과"""
    model_name: str
    elapsed_s: float

    # PyTorch reference
    ref_output_shape: list
    ref_output_dtype: str

    # 변환 전 (linalg)
    ops_before: int
    ops_before_list: list

    # 변환 후 (rngd.*)
    ops_after: int
    rngd_count: int
    rngd_ops: list             # (op_name, count) 리스트

    # 마커 분류
    marker_summary: dict

    # 데이터 흐름
    dataflow_nodes: int
    dataflow_edges: int        # rngd 간 내부 간선 수

    # fusion
    fusion_groups: list        # (name, count) 리스트

    # 검증 항목들
    checks: list = field(default_factory=list)

    @property
    def all_passed(self):
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self):
        return sum(1 for c in self.checks if c.passed)

    @property
    def fail_count(self):
        return sum(1 for c in self.checks if not c.passed)


def verify_transform(model, example_input, model_name="model"):
    """모델의 IR 변환을 자동 검증.

    Args:
        model: PyTorch nn.Module (eval 모드)
        example_input: 모델 입력 텐서 (단일 텐서)
        model_name: 보고서에 표시할 이름

    Returns:
        VerifyResult
    """
    t0 = time.time()
    checks = []

    # ── Step 1: PyTorch 직접 실행 → reference ──
    with torch.no_grad():
        ref_output = model(example_input)
    ref_shape = list(ref_output.shape)
    ref_dtype = str(ref_output.dtype)

    # ── Step 2: torch_mlir 컴파일 ──
    try:
        compiled = torch_mlir.compile(model, (example_input,),
                                       output_type="linalg-on-tensors", use_make_fx=True)
    except Exception as e:
        checks.append(VerifyCheck(
            name="torch_mlir 컴파일",
            passed=False,
            expected="컴파일 성공",
            actual=f"실패: {str(e)[:100]}",
        ))
        return VerifyResult(
            model_name=model_name, elapsed_s=time.time()-t0,
            ref_output_shape=ref_shape, ref_output_dtype=ref_dtype,
            ops_before=0, ops_before_list=[], ops_after=0, rngd_count=0,
            rngd_ops=[], marker_summary={}, dataflow_nodes=0, dataflow_edges=0,
            fusion_groups=[], checks=checks,
        )

    checks.append(VerifyCheck(
        name="torch_mlir 컴파일", passed=True,
        expected="컴파일 성공", actual="성공",
    ))

    # ── Step 2.5: RefBackend 수치 검증 (linalg IR → CPU 실행 → PyTorch 비교) ──
    # 중요: RefBackend.compile()이 Module을 소비하므로 별도 복사본 사용
    ir_text_for_ref = str(compiled)  # IR 텍스트를 먼저 저장
    try:
        from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend
        import numpy as np

        # 별도 Module 파싱하여 RefBackend에 전달
        ref_ctx = Context()
        ref_module = Module.parse(ir_text_for_ref, ref_ctx)
        backend = RefBackendLinalgOnTensorsBackend()
        lowered = backend.compile(ref_module)
        loaded = backend.load(lowered)
        # RefBackend 입력 형식: numpy array
        input_np = example_input.numpy()
        backend_output = loaded.forward(input_np)

        # numpy → torch로 변환해서 비교
        if isinstance(backend_output, np.ndarray):
            backend_tensor = torch.from_numpy(backend_output)
        else:
            backend_tensor = torch.tensor(backend_output)

        # 수치 비교
        max_diff = (ref_output - backend_tensor).abs().max().item()
        mean_diff = (ref_output - backend_tensor).abs().mean().item()
        ref_norm = ref_output.abs().max().item()
        rel_err = max_diff / (ref_norm + 1e-10)

        checks.append(VerifyCheck(
            name="linalg IR 수치 정확성 (RefBackend)",
            passed=rel_err < 1e-4,
            expected="상대 오차 < 0.01%",
            actual=f"상대 오차 {rel_err*100:.6f}% (max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f})",
            detail="PyTorch 직접 실행 결과와 linalg IR의 RefBackend CPU 실행 결과를 수치 비교. 이 검증이 통과하면 torch_mlir이 생성한 IR이 원본 모델과 동일한 값을 계산함을 의미.",
        ))
    except Exception as e:
        checks.append(VerifyCheck(
            name="linalg IR 수치 정확성 (RefBackend)",
            passed=False,
            expected="수치 비교 성공",
            actual=f"실행 실패: {str(e)[:150]}",
            detail="RefBackend로 linalg IR을 CPU에서 실행하여 PyTorch 결과와 비교. 실패 시 torch_mlir 컴파일 또는 RefBackend 호환성 문제.",
        ))

    # ── Step 3: 변환 전 스냅샷 ──
    ir_before = ir_text_for_ref if 'ir_text_for_ref' in dir() else str(compiled)
    ctx = Context()
    module = Module.parse(ir_before, ctx)
    ops_before = _list_op_names(module.operation)
    ops_before_counter = Counter(ops_before)

    # 변환 전 최종 출력 shape 추출
    pre_output_shape = _extract_return_shape(module)

    # ── Step 4: rewrite_to_rngd ──
    try:
        specs, _, _ = rewrite_to_rngd(module)
        checks.append(VerifyCheck(
            name="rewrite_to_rngd 실행", passed=True,
            expected="변환 성공", actual=f"{len(specs)}개 rngd.* op 생성",
        ))
    except Exception as e:
        checks.append(VerifyCheck(
            name="rewrite_to_rngd 실행", passed=False,
            expected="변환 성공", actual=f"실패: {str(e)[:100]}",
        ))
        return VerifyResult(
            model_name=model_name, elapsed_s=time.time()-t0,
            ref_output_shape=ref_shape, ref_output_dtype=ref_dtype,
            ops_before=len(ops_before), ops_before_list=ops_before_counter.most_common(),
            ops_after=0, rngd_count=0, rngd_ops=[],
            marker_summary={}, dataflow_nodes=0, dataflow_edges=0,
            fusion_groups=[], checks=checks,
        )

    # ── Step 5: 변환 후 스냅샷 ──
    ops_after = _list_op_names(module.operation)
    ops_after_counter = Counter(ops_after)
    rngd_counter = Counter(s.rngd_op for s in specs)

    # 마커 분류
    report = mark_remaining(module)

    # 데이터 흐름
    nodes = extract_dataflow(module)
    rngd_edges = sum(1 for n in nodes for s in n.input_sources if s == "rngd")

    # fusion
    groups = detect_fusion_patterns(nodes)
    fusion_counter = Counter(g.name for g in groups)

    # 최적화 마커: 식별 결과와 실행 위치를 IR 연산 속성으로 기록 (연산·데이터 흐름은 바꾸지 않음)
    insert_optimization_markers(module, groups)

    # 변환 후 최종 출력 shape 추출
    post_output_shape = _extract_return_shape(module)

    # ── Step 6: 검증 항목 실행 ──

    # 검증 1: 최종 출력 shape 보존
    checks.append(VerifyCheck(
        name="출력 shape 보존",
        passed=ref_shape == post_output_shape if post_output_shape else False,
        expected=str(ref_shape),
        actual=str(post_output_shape),
        detail="PyTorch 출력 shape과 변환 후 IR 출력 shape이 동일해야 함",
    ))

    # 검증 2: 출력 shape 변환 전후 일관성
    checks.append(VerifyCheck(
        name="변환 전후 출력 shape 일관성",
        passed=pre_output_shape == post_output_shape,
        expected=str(pre_output_shape),
        actual=str(post_output_shape),
        detail="rewrite가 IR의 최종 출력 shape을 변경하지 않아야 함",
    ))

    # 검증 3: rngd.* op이 1개 이상 생성
    checks.append(VerifyCheck(
        name="rngd.* op 생성",
        passed=len(specs) > 0,
        expected="> 0",
        actual=str(len(specs)),
        detail="최소 1개 이상의 연산이 rngd.* IR로 변환되어야 함",
    ))

    # 검증 4: CPU_FALLBACK 비율
    cpu_fallback = len(report.get("CPU_FALLBACK", []))
    total_compute = len(specs) + cpu_fallback + sum(1 for n, r in report.get("UNKNOWN", []) if "linalg.generic" in n)
    coverage = len(specs) / max(total_compute, 1) * 100
    checks.append(VerifyCheck(
        name="연산 변환 커버리지",
        passed=coverage >= 80.0,
        expected="≥ 80%",
        actual=f"{coverage:.1f}% ({len(specs)}/{total_compute})",
        detail="계산 op 중 80% 이상이 rngd.*로 변환되어야 함",
    ))

    # 검증 5: 데이터 흐름 연결성 (rngd 간 내부 간선이 존재)
    checks.append(VerifyCheck(
        name="데이터 흐름 연결성",
        passed=rngd_edges >= len(specs) * 0.3,
        expected=f"≥ {int(len(specs) * 0.3)} 간선",
        actual=f"{rngd_edges} 간선",
        detail="rngd.* op 간 내부 데이터 의존 관계가 충분히 존재해야 함 (고립된 op이 아님)",
    ))

    # 검증 6: 복합 패턴 식별
    checks.append(VerifyCheck(
        name="복합 패턴 식별",
        passed=len(groups) > 0,
        expected="> 0 그룹",
        actual=f"{len(groups)}개 그룹 ({dict(fusion_counter)})",
        detail="RMSNorm, Softmax 등 복합 패턴이 최소 1개 이상 감지되어야 함",
    ))

    # 검증 7: 각 rngd.* op의 출력 shape 유효성
    invalid_shapes = []
    for n in nodes:
        if not n.out_shape or any(d <= 0 for d in n.out_shape):
            invalid_shapes.append(f"{n.ssa_name}: {n.out_shape}")
    checks.append(VerifyCheck(
        name="rngd.* op 출력 shape 유효성",
        passed=len(invalid_shapes) == 0,
        expected="모든 op의 출력 shape > 0",
        actual=f"{len(invalid_shapes)}개 비정상" if invalid_shapes else "전부 유효",
        detail="; ".join(invalid_shapes[:5]) if invalid_shapes else "",
    ))

    # 검증 8: dtype 일관성 (f32 연산 체인 내에서 갑자기 bf16이 섞이지 않는지)
    dtype_issues = []
    for n in nodes:
        if n.op_type == "rngd.elementwise" and n.dtype not in ("f32", "f64"):
            dtype_issues.append(f"{n.ssa_name}: elementwise인데 dtype={n.dtype}")
        if n.op_type in ("rngd.batch_gemm", "rngd.gemm") and n.dtype not in ("f32", "bf16", "f16"):
            dtype_issues.append(f"{n.ssa_name}: gemm인데 dtype={n.dtype}")
    checks.append(VerifyCheck(
        name="dtype 일관성",
        passed=len(dtype_issues) == 0,
        expected="각 op family에 맞는 dtype",
        actual=f"{len(dtype_issues)}개 이상" if dtype_issues else "전부 일관",
        detail="; ".join(dtype_issues[:5]) if dtype_issues else "",
    ))

    # 검증 9: UNKNOWN op 비율
    unknown_count = len(report.get("UNKNOWN", []))
    checks.append(VerifyCheck(
        name="미분류(UNKNOWN) op 비율",
        passed=unknown_count <= len(ops_after) * 0.1,
        expected=f"≤ {int(len(ops_after) * 0.1)}개 (전체의 10%)",
        actual=f"{unknown_count}개",
        detail="분류되지 않은 op이 많으면 파이프라인 커버리지에 빈틈이 있음",
    ))

    elapsed = time.time() - t0

    return VerifyResult(
        model_name=model_name,
        elapsed_s=round(elapsed, 2),
        ref_output_shape=ref_shape,
        ref_output_dtype=ref_dtype,
        ops_before=len(ops_before),
        ops_before_list=ops_before_counter.most_common(),
        ops_after=len(ops_after),
        rngd_count=len(specs),
        rngd_ops=rngd_counter.most_common(),
        marker_summary={k: len(v) for k, v in report.items()},
        dataflow_nodes=len(nodes),
        dataflow_edges=rngd_edges,
        fusion_groups=fusion_counter.most_common(),
        checks=checks,
    )


def _extract_return_shape(module):
    """MLIR Module에서 func.return의 operand shape 추출"""
    try:
        for op in module.operation.regions[0].blocks[0].operations:
            for region in op.regions:
                for block in region.blocks:
                    for inner in block.operations:
                        if str(inner.name) == "func.return" and inner.operands:
                            return list(RankedTensorType(inner.operands[0].type).shape)
    except:
        pass
    return None


def print_verify_report(result):
    """검증 결과를 터미널에 출력"""
    print(f"\n{'='*60}")
    print(f"=== IR 변환 정확성 검증: {result.model_name} ===")
    print(f"    소요시간: {result.elapsed_s}s")
    print(f"{'='*60}")

    print(f"\n  PyTorch reference:")
    print(f"    출력 shape: {result.ref_output_shape}")
    print(f"    출력 dtype: {result.ref_output_dtype}")

    print(f"\n  변환 결과:")
    print(f"    변환 전: {result.ops_before}개 op")
    print(f"    변환 후: {result.ops_after}개 op")
    print(f"    rngd.*: {result.rngd_count}개")
    print(f"    dataflow: {result.dataflow_nodes}개 노드, {result.dataflow_edges}개 내부 간선")
    print(f"    fusion: {dict(result.fusion_groups)}")

    print(f"\n  마커 분류:")
    for kind, count in sorted(result.marker_summary.items()):
        print(f"    {kind}: {count}개")

    print(f"\n{'─'*60}")
    print(f"  검증 결과: {result.pass_count} PASS / {result.fail_count} FAIL (총 {len(result.checks)}개)")
    print(f"{'─'*60}")

    for c in result.checks:
        icon = "✅" if c.passed else "❌"
        print(f"\n  {icon} {c.name}")
        print(f"     기대: {c.expected}")
        print(f"     실제: {c.actual}")
        if c.detail:
            print(f"     상세: {c.detail}")

    print(f"\n{'='*60}")
    if result.all_passed:
        print(f"  ✅ 전체 PASS ({result.pass_count}/{len(result.checks)})")
    else:
        print(f"  ❌ {result.fail_count}개 FAIL")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════
# Phase 2: NPU 시뮬레이터 E2E 검증
# 기존 Rust 커널 E2E 테스트를 subprocess로 실행하여 결과 수집.
# verify_transform의 Level 3 검증.
# ══════════════════════════════════════════════════════════════════

import subprocess, re

E2E_TESTS = [
    {
        "name": "FFN E2E",
        "bin": "pilot_golden_ffn_e2e",
        "module": "FFN",
        "desc": "gate→SiLU→up→gate·up→down 전체 체인을 NPU 시뮬레이터에서 실행하여 PyTorch 기준값과 비교",
    },
    {
        "name": "RMSNorm E2E",
        "bin": "pilot_golden_rmsnorm_e2e",
        "module": "RMSNorm",
        "desc": "pow2→reduce_sum(트리)→rsqrt→mul 체인을 NPU 시뮬레이터에서 실행하여 PyTorch 기준값과 비교",
    },
    {
        "name": "Attention E2E",
        "bin": "pilot_golden_attn_e2e",
        "module": "Attention",
        "desc": "QKV proj→score(QK^T)→AV→O proj 체인을 NPU 시뮬레이터에서 실행하여 PyTorch 기준값과 비교",
    },
    {
        "name": "Full Block E2E",
        "bin": "pilot_golden_block_e2e",
        "module": "Block",
        "desc": "RMSNorm→Attention→Residual→RMSNorm→FFN→Residual 전체 Decoder Block을 NPU 시뮬레이터에서 실행",
    },
]


@dataclass
class E2EResult:
    """단일 E2E 테스트 결과"""
    name: str
    module: str
    passed: bool
    max_diff: float
    mean_diff: float
    log_tail: str
    desc: str
    elapsed_s: float


def run_e2e_tests(timeout=120) -> list:
    """모든 E2E 시뮬레이터 테스트를 실행하고 결과 수집"""
    results = []
    for test in E2E_TESTS:
        t0 = time.time()
        try:
            from .kernel_backend import run_kernel as _backend_run
            passed, log, _stats = _backend_run(test['bin'], timeout)

            # diff 파싱
            diffs = re.findall(r"diff=([\d.e+-]+)", log)
            max_diff = max(float(d) for d in diffs) if diffs else -1.0
            mean_diff = sum(float(d) for d in diffs) / len(diffs) if diffs else -1.0

            results.append(E2EResult(
                name=test["name"], module=test["module"],
                passed=passed, max_diff=round(max_diff, 6), mean_diff=round(mean_diff, 6),
                log_tail=log[-300:] if not passed else "",
                desc=test["desc"], elapsed_s=round(time.time()-t0, 1),
            ))
        except subprocess.TimeoutExpired:
            results.append(E2EResult(
                name=test["name"], module=test["module"],
                passed=False, max_diff=-1, mean_diff=-1,
                log_tail=f"TIMEOUT ({timeout}s)",
                desc=test["desc"], elapsed_s=timeout,
            ))
        except Exception as e:
            results.append(E2EResult(
                name=test["name"], module=test["module"],
                passed=False, max_diff=-1, mean_diff=-1,
                log_tail=str(e)[:200],
                desc=test["desc"], elapsed_s=round(time.time()-t0, 1),
            ))
    return results


def print_e2e_report(results: list):
    """E2E 시뮬레이터 검증 결과 출력"""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"=== NPU 시뮬레이터 E2E 검증 ({passed}/{total} PASS) ===")
    print(f"{'='*60}")

    for r in results:
        icon = "✅" if r.passed else "❌"
        diff_str = f"max_diff={r.max_diff}" if r.max_diff >= 0 else "측정 불가"
        print(f"\n  {icon} {r.name} ({r.elapsed_s}s)")
        print(f"     {r.desc}")
        print(f"     결과: {diff_str}")
        if r.log_tail:
            print(f"     로그: {r.log_tail[:100]}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import torch.nn as nn

    from .models import LlamaDecoderLayer

    # 0.6 스케일 테스트
    model = LlamaDecoderLayer(64, 128, 8, 2, 8, 8)
    model.eval()
    x = torch.randn(1, 8, 64)

    result = verify_transform(model, x, "LlamaDecoderLayer(dim=64, GQA+RoPE)")
    print_verify_report(result)


    # 8B 스케일 테스트
    print("\n\n")
    model_8b = LlamaDecoderLayer(4096, 14336, 32, 8, 128, 8)
    model_8b.eval()
    x_8b = torch.randn(1, 8, 4096)

    result_8b = verify_transform(model_8b, x_8b, "LlamaDecoderLayer(dim=4096, GQA+RoPE) — 8B 스케일")
    print_verify_report(result_8b)

    # 0.6 스케일 vs 8B 비교
    print("\n=== 0.6 스케일 vs 8B 스케일 비교 ===")
    print(f"  {'항목':<25} {'0.6 스케일':>12} {'8B':>10} {'일치':>6}")
    print(f"  {'─'*55}")
    items = [
        ("rngd.* op 수", result.rngd_count, result_8b.rngd_count),
        ("복합 패턴 그룹 수", len(result.fusion_groups), len(result_8b.fusion_groups)),
        ("검증 PASS 수", result.pass_count, result_8b.pass_count),
        ("검증 FAIL 수", result.fail_count, result_8b.fail_count),
    ]
    for name, v1, v2 in items:
        match = "✅" if v1 == v2 else "⚠️"
        print(f"  {name:<25} {v1:>10} {v2:>10} {match:>6}")
