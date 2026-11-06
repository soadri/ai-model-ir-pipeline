"""
사용자 정의 모델 변환·분석 — 입력한 PyTorch 코드로 IR 변환, Level 1-2, Level 3를 수행합니다.

기존 verify_router.py의 custom_convert 흐름을 옮긴 것으로, 응답 JSON 구조는 그대로입니다.

보안 주의: 입력 코드는 이 프로세스 안에서 그대로 실행됩니다. 아래 키워드 검사는 실수를
막기 위한 최소한의 장치일 뿐 보안 경계가 아니므로, 이 서버는 사용자 본인의 PC(로컬 Docker)
에서만 실행해야 하며 외부에 공개된 서비스로 운영해서는 안 됩니다.
"""
from __future__ import annotations

import ast
import hashlib
import time

from server import storage
from server.services.preset_verify import filter_dense_constants

_BLOCKED = ["os.system", "subprocess", "shutil.rmtree", "open(", "__import__",
            "eval(", "exec(", "compile(", "globals(", "locals(",
            "sys.exit", "os.remove", "os.unlink", "os.rmdir",
            "pathlib", "socket", "requests", "urllib"]

# Level 3 참조값 — Llama-3.1 8B Decoder Layer 검증에서 연산 종류별로 측정한 오차
_NPU_REF = {
    "rngd.gemm": {"max_diff": 27.13, "rel_err_pct": 0.9, "source": "FFN 0.6 검증 (gemm M=512,K=64,N=512, 상대오차 0.9%)"},
    "rngd.batch_gemm": {"max_diff": 2.96, "rel_err_pct": 3.0, "source": "Attention 0.6 검증 (gemv I=256,J=2048, 상대오차 3.0%)"},
    "rngd.elementwise": {"max_diff": 0.0, "rel_err_pct": 0.0, "source": "RMSNorm 0.6 검증 (A=16, diff=0)"},
    "rngd.reduce": {"max_diff": 0.0, "rel_err_pct": 0.0, "source": "RMSNorm 0.6 검증 (NPU 트리 reduce, diff=0)"},
    "rngd.broadcast": {"max_diff": 0.0, "rel_err_pct": 0.0, "source": "데이터 확장 (오차 없음)"},
}


class InputError(Exception):
    """사용자 입력 코드의 문제 — 메시지를 그대로 화면에 보여줍니다."""


def prepare(code: str):
    """입력 코드를 검사·실행해 (model, x)를 돌려줍니다."""
    if not code.strip():
        raise InputError("코드가 비어있습니다.")
    lowered = code.lower()
    for word in _BLOCKED:
        if word.lower() in lowered:
            raise InputError(f"보안: '{word}' 사용이 제한됩니다. PyTorch nn.Module 정의와 텐서 생성만 허용됩니다.")
    if "nn.Module" not in code and "nn.Sequential" not in code:
        raise InputError("nn.Module 또는 nn.Sequential을 상속하는 모델 클래스가 필요합니다.")
    if "model" not in code:
        raise InputError("model 변수가 정의되어야 합니다. 예: model = MyModel()")

    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 — 로컬 실행 전용 (모듈 설명 참고)
    model, x = namespace.get("model"), namespace.get("x")
    if model is None or x is None:
        raise InputError("코드에 model과 x 변수가 정의되어야 합니다.")
    model.eval()
    return model, x


def _level12(model, x) -> dict:
    """RefBackend로 linalg IR을 실행해 PyTorch 출력과 비교"""
    try:
        import numpy as np
        import torch_mlir
        from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

        backend = RefBackendLinalgOnTensorsBackend()
        compiled = torch_mlir.compile(model, (x,), output_type="linalg-on-tensors", use_make_fx=True)
        loaded = backend.load(backend.compile(compiled))
        ref_out = loaded.forward(x.numpy())
        pt_out = model(x).detach().numpy()
        max_diff = float(np.max(np.abs(ref_out - pt_out)))
        mean_diff = float(np.mean(np.abs(ref_out - pt_out)))
        rel_err = max_diff / max(float(np.max(np.abs(pt_out))), 1e-10) * 100
        return {"passed": max_diff < 0.01, "max_diff": round(max_diff, 10), "mean_diff": round(mean_diff, 10),
                "rel_err_pct": round(rel_err, 6), "output_shape": list(pt_out.shape)}
    except Exception as e:  # noqa: BLE001
        return {"passed": False, "error": str(e)}


def _group_unsupported(report: dict) -> list[dict]:
    """미지원 연산을 복합 연산 단위로 묶습니다 (generic 내부 연산의 중복 집계 제거)."""
    groups, body_ops_set = [], set()
    for name, reason in report.get("UNKNOWN", []):
        if name == "linalg.generic" and "미분류 body:" in reason:
            body_str = reason.split("미분류 body: ")[1]
            try:
                body_ops = ast.literal_eval(body_str)
            except Exception:  # noqa: BLE001
                body_ops = [body_str]
            body = set(body_ops)
            if "math.erf" in body:
                label = "GELU"
            elif "math.tanh" in body:
                label = "GELU(tanh 근사)" if "math.exp" not in body else "Tanh 복합"
            elif {"math.exp", "arith.addf", "arith.divf"} <= body:
                label = "Sigmoid"
            elif "math.exp" in body:
                label = "Exp 복합"
            elif "math.log" in body:
                label = "Log 복합"
            else:
                label = "복합 연산"
            groups.append({"type": "group", "label": label, "body": body_ops,
                           "reason": f"linalg.generic 내부에 {len(body_ops)}개 연산으로 구성된 복합 연산. 현재 변환 규칙 미등록."})
            body_ops_set.update(body_ops)
        else:
            groups.append({"type": "single", "name": name, "reason": reason})

    for name, reason in report.get("CPU_FALLBACK", []):
        if name not in body_ops_set:
            groups.append({"type": "single", "name": name, "reason": reason})
        else:
            body_ops_set.discard(name)  # 이미 그룹에 포함된 연산은 한 번만 차감
    return groups


def _npu_refs(report: dict) -> tuple[list[dict], bool]:
    used: dict[str, int] = {}
    for name, _ in report.get("CONVERTED", []):
        used[name] = used.get(name, 0) + 1
    refs, all_verified = [], True
    for op, count in sorted(used.items()):
        ref = _NPU_REF.get(op)
        if ref:
            refs.append({"op": op, "count": count, "max_diff": ref["max_diff"],
                         "rel_err_pct": ref.get("rel_err_pct"), "source": ref["source"], "verified": True})
        else:
            refs.append({"op": op, "count": count, "max_diff": -1, "source": "미검증", "verified": False})
            all_verified = False
    return refs, all_verified


def analyze(model, x, code: str, started: float) -> dict:
    """IR 변환과 3단계 검증. 호출하는 쪽에서 실행 잠금을 잡고 있어야 합니다."""
    import torch_mlir
    from torch_mlir.ir import Context, Module
    from rngd_pipeline.e2e_pipeline import (detect_fusion_patterns, extract_dataflow,
                                            mark_remaining, rewrite_to_rngd)

    compiled = torch_mlir.compile(model, (x,), output_type="linalg-on-tensors", use_make_fx=True)
    module = Module.parse(str(compiled), Context())
    rewrite_to_rngd(module)

    level12 = _level12(model, x)
    report = mark_remaining(module)

    fusion_dict: dict[str, int] = {}
    try:
        for fg in detect_fusion_patterns(extract_dataflow(module)):
            fusion_dict[fg.name] = fusion_dict.get(fg.name, 0) + 1
    except Exception:  # noqa: BLE001
        fusion_dict = {}

    groups = _group_unsupported(report)
    converted = len(report.get("CONVERTED", []))
    total_compute = converted + len(report.get("CPU_FALLBACK", [])) + len(report.get("UNKNOWN", []))
    coverage = round(converted / max(total_compute, 1) * 100, 1)
    marker_summary = {k: len(v) for k, v in report.items() if v}

    linalg_ir, rngd_ir = str(compiled), str(module)
    refs, all_verified = _npu_refs(report)

    try:
        from rngd_pipeline.custom_npu_verify import verify_custom_model_full
        level3 = verify_custom_model_full(model, x)
    except Exception as e:  # noqa: BLE001
        level3 = {"error": str(e), "summary": {"total_linears": 0, "tested": 0, "passed": 0}}

    result = {
        "converted_count": converted,
        "coverage_pct": coverage,
        "unsupported": groups,
        "unsupported_count": len(groups),
        "markers": marker_summary,
        "fusion": fusion_dict,
        "level12": level12,
        "npu_ref": {"refs": refs, "all_verified": all_verified,
                    "basis": "Llama-3.1 8B Decoder Layer (furiosa-opt 0.6, gemm M=512 현실적 규모)"},
        "elapsed_s": round(time.time() - started, 2),
        "ir": {
            "linalg_preview": "\n".join(
                line if "dense<" not in line else line[: line.index("dense<")] + "dense<...> (가중치 상수 생략)"
                for line in linalg_ir.split("\n")
            )[:3000],
            "linalg_lines": linalg_ir.count("\n"),
            "rngd_preview": rngd_ir[:2000],
            "rngd_lines": rngd_ir.count("\n"),
            "rngd_ops": [line.strip() for line in rngd_ir.split("\n") if "rngd." in line],
            "rngd_ir_full": filter_dense_constants(rngd_ir),
            "linalg_ir_full": filter_dense_constants(linalg_ir),
        },
        "marker_details": {k: [{"name": n, "reason": r} for n, r in v] for k, v in report.items() if v},
        "level3": level3,
    }

    summary = level3.get("summary", {}) if isinstance(level3, dict) else {}
    storage.save_run({
        "model_name": "커스텀 모델",
        "model_type": "custom",
        "model_sub_type": "user_defined",
        "model_code_hash": hashlib.md5(code.encode()).hexdigest()[:12],
        "model_code": code[:5000],
        "coverage_pct": coverage,
        "converted_count": converted,
        "total_count": converted + len(groups),
        "pass_l12": 1 if level12.get("passed") else 0,
        "total_l12": 1,
        "pass_l3": summary.get("passed", 0),
        "total_l3": summary.get("tested", 0),
        "fusion": fusion_dict,
        "markers": marker_summary,
        "rngd_ir": filter_dense_constants(rngd_ir),
        "linalg_ir": filter_dense_constants(linalg_ir),
    })
    return result
