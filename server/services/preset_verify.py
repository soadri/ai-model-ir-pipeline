"""
기본 모델(Llama-3.1 8B Decoder Layer) 통합 검증 — Level 1-2, Level 3, IR 미리보기.

기존 verify_router.py의 ir_verify / npu_verify / _do_all_verify 흐름을 한곳에 모은 것으로,
응답 JSON 구조와 각 단계의 계산 방식은 그대로 유지합니다. 바뀐 점은 다음과 같습니다.

  - 경로·커널 실행은 config / kernel_backend를 통해서만 접근
  - 저장하는 커널 소스는 실제로 실행한 커널 디렉토리(config.KERNEL_DIR)에서 읽음
  - 스냅샷 비교는 RNGD_SNAPSHOT_DIR가 지정된 경우에만 수행
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from rngd_pipeline import config
from server import storage

# 검증 규모별 모델 구성 (dim, hidden, n_heads, n_kv_heads, head_dim, seq)
SCALES = {
    "sim_small": dict(dim=16, hidden=8, n_heads=1, n_kv_heads=1, head_dim=16, seq=8),
    "sim_mid": dict(dim=64, hidden=128, n_heads=8, n_kv_heads=2, head_dim=8, seq=8),
    "sim_full": dict(dim=4096, hidden=14336, n_heads=32, n_kv_heads=8, head_dim=128, seq=8),
}
DEFAULT_SCALE = "sim_small"

# Level 3 결과와 함께 저장하는 커널: 모듈명 → 검증 프로파일의 커널 키
_KERNEL_MODULES = [("RMSNorm", "rmsnorm"), ("FFN", "ffn"), ("Attention", "attn"), ("FullBlock", "block")]

_SNAPSHOT_DIR = os.environ.get("RNGD_SNAPSHOT_DIR")


def _scale(scale: str) -> dict:
    return SCALES.get(scale, SCALES[DEFAULT_SCALE])


def filter_dense_constants(ir_text: str) -> str:
    """IR 텍스트에서 dense<...> 가중치 상수를 생략해 구조만 남깁니다."""
    out = []
    for line in ir_text.splitlines():
        if "dense<" in line:
            out.append(line[: line.index("dense<")] + "dense<...>")
        else:
            out.append(line)
    return "\n".join(out)


def _compile_linalg(model, x):
    import torch_mlir
    return torch_mlir.compile(model, (x,), output_type="linalg-on-tensors", use_make_fx=True)


# ── Level 1-2 ─────────────────────────────────────────────────────

def _marker_details(model, x) -> dict:
    """마커 분류별 상세 내역 (카테고리별 op와 사유)"""
    from torch_mlir.ir import Context, Module
    from rngd_pipeline.e2e_pipeline import mark_remaining, rewrite_to_rngd
    module = Module.parse(str(_compile_linalg(model, x)), Context())
    rewrite_to_rngd(module)
    return mark_remaining(module)


def run_level12(scale: str = DEFAULT_SCALE) -> dict:
    """IR 변환 정합성 검증 (Level 1-2)"""
    import torch
    from rngd_pipeline.models import LlamaDecoderLayer
    from rngd_pipeline.verify_transform import verify_transform

    sd = _scale(scale)
    # 기존 대시보드 기준 모델 구성 유지: seq 길이만큼의 RoPE 버퍼, n_rep=1이어도 KV 반복
    model = LlamaDecoderLayer(sd["dim"], sd["hidden"], sd["n_heads"], sd["n_kv_heads"],
                              sd["head_dim"], sd["seq"], always_repeat_kv=True)
    model.eval()
    x = torch.randn(1, sd["seq"], sd["dim"])
    model_name = (f"LlamaDecoderLayer(dim={sd['dim']}, hidden={sd['hidden']}, n_heads={sd['n_heads']}, "
                  f"n_kv_heads={sd['n_kv_heads']}, GQA+RoPE)")
    result = verify_transform(model, x, model_name)

    return {
        "model": result.model_name,
        "elapsed_s": result.elapsed_s,
        "ref_output_shape": result.ref_output_shape,
        "ref_output_dtype": result.ref_output_dtype,
        "ops_before": result.ops_before,
        "ops_after": result.ops_after,
        "rngd_count": result.rngd_count,
        "dataflow_nodes": result.dataflow_nodes,
        "dataflow_edges": result.dataflow_edges,
        "fusion": dict(result.fusion_groups),
        "markers": result.marker_summary,
        "marker_samples": {k: [(name, reason) for name, reason in v[:3]]
                           for k, v in _marker_details(model, x).items()},
        "all_passed": result.all_passed,
        "pass_count": result.pass_count,
        "fail_count": result.fail_count,
        "checks": [
            {"name": c.name, "passed": c.passed, "expected": c.expected,
             "actual": c.actual, "detail": c.detail}
            for c in result.checks
        ],
    }


# ── Level 3 ───────────────────────────────────────────────────────

def _run_level3_full_scale() -> dict:
    """sim_full: 실규모 Decoder Layer 커널 단일 실행"""
    from rngd_pipeline.kernel_backend import run_kernel

    passed, output, _ = run_kernel("pilot_golden_block_dim4096f", timeout=600)
    if output == "TIMEOUT":
        return {"error": "TIMEOUT (600s)", "profile": "sim_full", "all_passed": False}

    def num(pattern, cast=float, default=-1):
        m = re.search(pattern, output)
        return cast(m.group(1)) if m else default

    samples = [
        {"idx": int(m.group(1)), "expected": float(m.group(2)),
         "actual": float(m.group(3)), "diff": float(m.group(4))}
        for m in re.finditer(r"\[(\d+)\] exp=([\-\d.]+) act=([\-\d.]+) diff=([\d.]+)", output)
    ]
    scale_info = {
        "max_diff": num(r"max_diff=([\d.]+)"),
        "rel_err_pct": num(r"rel_err=([\d.]+)%"),
        "gemm_count": num(r"총 GEMM: (\d+)회", int, 0),
        "elapsed_s": num(r"elapsed: ([\d.]+)s", float, 0),
        "samples": samples,
        "features": {"rope": True, "causal_mask": True, "k_tiling": "K=256", "m_kernel": "M=16"},
    }
    return {
        "profile": "sim_full",
        "profile_desc": "Llama-3.1 8B 실스케일 (dim=4096, RoPE+CausalMask)",
        "all_passed": passed,
        "scale_info": scale_info,
        "checks": [{
            "module": "Full Decoder Layer (dim=4096)",
            "fusion": "FULL_DECODER",
            "e2e_bin": "pilot_golden_block_dim4096f",
            "passed": passed,
            "max_diff": scale_info["max_diff"],
            "rel_err_pct": scale_info["rel_err_pct"],
            "pytorch_shape": [1, 8, 4096],
            "detail": (f"GEMM {scale_info['gemm_count']}회, RoPE+CausalMask, "
                       f"rel_err={scale_info['rel_err_pct']:.2f}%"),
            "elapsed_s": scale_info["elapsed_s"],
            "data_injected": True,
        }],
        "elapsed_s": scale_info["elapsed_s"],
    }


def run_level3(scale: str = DEFAULT_SCALE) -> dict:
    """NPU 시뮬레이터 연산 정합성 검증 (Level 3)"""
    if scale == "sim_full":
        return _run_level3_full_scale()

    from rngd_pipeline.npu_direct_verify import run_full_verify
    report = run_full_verify(scale)
    return {
        "profile": report.profile_name,
        "profile_desc": report.profile_desc,
        "ir": {
            "rngd_count": report.ir_result.rngd_count,
            "coverage_pct": report.ir_result.coverage_pct,
            "fusion": report.ir_result.fusion_groups,
        },
        "captures": report.captures_info,
        "checks": [
            {"module": r.module_name, "fusion": r.fusion_group, "e2e_bin": r.e2e_bin,
             "pytorch_shape": r.pytorch_output_shape, "max_diff": r.max_diff, "passed": r.passed,
             "detail": r.detail, "elapsed_s": r.elapsed_s, "data_injected": r.data_injected,
             "rel_err_pct": r.rel_err_pct, "judgment": r.judgment}
            for r in report.module_results
        ],
        "all_passed": report.all_passed,
        "pass_count": report.pass_count,
        "total": len(report.module_results),
        "elapsed_s": report.total_elapsed_s,
    }


# ── 스냅샷 비교 (선택) ─────────────────────────────────────────────

def _snapshot_comparison() -> dict | None:
    """정기 검증 스냅샷이 있는 환경에서만 직전 대비 변화를 비교합니다."""
    if not _SNAPSHOT_DIR:
        return None
    snapshots = sorted(Path(_SNAPSHOT_DIR).glob("verify_*.json"))
    if len(snapshots) < 2:
        return None
    curr = json.loads(snapshots[-1].read_text())
    prev = json.loads(snapshots[-2].read_text())
    prev_mods = {m["name"]: m for m in prev["level3"]["modules"]}
    diffs = []
    for m in curr["level3"]["modules"]:
        pm = prev_mods.get(m["name"])
        if pm and pm["max_diff"] >= 0 and m["max_diff"] >= 0:
            ratio = m["max_diff"] / max(pm["max_diff"], 1e-10)
            diffs.append({"name": m["name"], "prev": pm["max_diff"], "curr": m["max_diff"],
                          "ratio": round(ratio, 2), "ok": ratio <= 2.0})
    return {
        "prev_timestamp": prev["timestamp"], "curr_timestamp": curr["timestamp"],
        "l12_prev": prev["level12"]["pass_count"], "l12_curr": curr["level12"]["pass_count"],
        "modules": diffs, "all_ok": all(d["ok"] for d in diffs),
    }


# ── IR 미리보기 ───────────────────────────────────────────────────

def _ir_detail(scale: str):
    """IR 미리보기, rngd.* 연산 목록, 마커 상세. 실패하면 (None, None)."""
    try:
        import torch
        from torch_mlir.ir import Context, Module
        from rngd_pipeline.e2e_pipeline import insert_optimization_markers, mark_remaining, rewrite_to_rngd
        from rngd_pipeline.models import LlamaDecoderLayer

        torch.manual_seed(42)
        sd = _scale(scale)
        model = LlamaDecoderLayer(sd["dim"], sd["hidden"], sd["n_heads"], sd["n_kv_heads"],
                                  sd["head_dim"], sd["seq"])
        model.eval()
        x = torch.randn(1, sd["seq"], sd["dim"])
        compiled = _compile_linalg(model, x)
        print(f"[all_verify] IR compile done for scale={scale}, dim={sd['dim']}")

        module = Module.parse(str(compiled), Context())
        linalg_ir = str(module)
        rewrite_to_rngd(module)
        insert_optimization_markers(module)
        rngd_ir = str(module)
        rngd_ops = [line.strip() for line in rngd_ir.split("\n") if "rngd." in line]
        report = mark_remaining(module)

        marker_details = {k: [{"name": n, "reason": r} for n, r in v] for k, v in report.items() if v}
        converted = len(report.get("CONVERTED", []))
        fallback = len(report.get("CPU_FALLBACK", [])) + len(report.get("UNKNOWN", []))
        ir_data = {
            "linalg_preview": "\n".join(
                line if "dense<" not in line
                else line[: line.index("dense<")] + "dense<...> (가중치 상수 생략)"
                for line in linalg_ir.split("\n")
            )[:3000],
            "linalg_lines": linalg_ir.count("\n"),
            "rngd_preview": rngd_ir[:2000],
            "rngd_ir_full": filter_dense_constants(rngd_ir),
            "linalg_ir_full": filter_dense_constants(linalg_ir),
            "rngd_lines": rngd_ir.count("\n"),
            "rngd_ops": rngd_ops,
            "rngd_count": len(rngd_ops),
            "coverage_pct": 100.0 if not fallback else round(converted / max(converted + fallback, 1) * 100, 1),
            "converted_count": converted,
            "markers_summary": {k: len(v) for k, v in report.items() if v},
        }
        return ir_data, marker_details
    except Exception:  # noqa: BLE001 — 미리보기 실패가 검증 결과를 막지 않도록
        return None, None


# ── 실행 이력 저장 ─────────────────────────────────────────────────

def _kernel_files(scale: str) -> list[tuple[str, Path]]:
    """검증 규모에서 실제로 실행한 커널의 소스 파일 (Cargo.toml의 [[bin]] 경로 기준)"""
    import re as _re
    from rngd_pipeline.npu_direct_verify import VERIFY_PROFILES

    cargo = config.KERNEL_DIR / "Cargo.toml"
    bin_paths = dict(_re.findall(r'\[\[bin\]\]\s*name\s*=\s*"([^"]+)"\s*path\s*=\s*"([^"]+)"',
                                 cargo.read_text())) if cargo.is_file() else {}
    kernels = (VERIFY_PROFILES.get(scale) or {}).get("kernels") or {}
    files = []
    for module_name, key in _KERNEL_MODULES:
        cfg = kernels.get(key)
        if cfg and cfg.get("bin") in bin_paths:
            files.append((module_name, config.KERNEL_DIR / bin_paths[cfg["bin"]]))
    return files


def _kernel_sources(level3: dict, scale: str = DEFAULT_SCALE) -> list[dict]:
    checks = (level3.get("checks") or []) if level3 else []
    by_module = {}
    for ch in checks:
        name = ch.get("module", "")
        if "RMSNorm" in name and "#1" in name:
            by_module["RMSNorm"] = ch
        elif "FFN" in name:
            by_module["FFN"] = ch
        elif "Attention" in name and "Block" not in name:
            by_module["Attention"] = ch
        elif "Block" in name:
            by_module["FullBlock"] = ch

    kernels = []
    for module_name, path in _kernel_files(scale):
        if not path.is_file():
            continue
        ch = by_module.get(module_name, {})
        kernels.append({"module_name": module_name, "source_code": path.read_text(encoding="utf-8"),
                        "max_diff": ch.get("max_diff", 0), "passed": ch.get("passed", True)})
    return kernels


def _save_run(level12: dict, level3: dict, ir_data: dict | None, scale: str = DEFAULT_SCALE) -> None:
    l12, l3, ir = level12 or {}, level3 or {}, ir_data or {}
    storage.save_run({
        "model_name": "Llama-3.1 8B (Decoder Layer)",
        "model_type": "preset",
        "model_arch": "Decoder-only (GQA+RoPE)",
        "model_params": "8B",
        "verify_scale": "furiosa-opt 0.6 (gemm M=512, elementwise A=16)",
        "core_ops": "Attention,Linear,Norm",
        "coverage_pct": ir.get("coverage_pct", 0),
        "converted_count": ir.get("rngd_count", 0),
        "total_count": ir.get("rngd_count", 0),
        "pass_l12": l12.get("pass_count", 0),
        "total_l12": len(l12.get("checks", [])),
        "pass_l3": l3.get("pass_count", 0),
        "total_l3": l3.get("total", 0),
        "fusion": ir.get("fusion", {}),
        "markers": ir.get("markers_summary", l12.get("markers", {})),
        "rngd_ir": ir.get("rngd_ir_full", ir.get("rngd_preview", "")),
        "linalg_ir": ir.get("linalg_ir_full", ir.get("linalg_preview", "")),
        "kernels": _kernel_sources(level3, scale),
    })


# ── 캐시 ──────────────────────────────────────────────────────────

def _cache_path(scale: str) -> Path:
    if scale not in SCALES:
        raise ValueError(f"알 수 없는 검증 규모: {scale}")
    return config.CACHE_DIR / f"verify_{scale}.json"


def save_cache(scale: str, data: dict) -> None:
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data["_cached_at"] = datetime.now().isoformat()
        _cache_path(scale).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def load_cache(scale: str) -> dict | None:
    try:
        path = _cache_path(scale)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return None


# ── 통합 실행 ─────────────────────────────────────────────────────

def run_all(scale: str = DEFAULT_SCALE) -> dict:
    """Level 1-2 → Level 3 → IR 미리보기 → 이력 저장 → 캐시"""
    level12 = run_level12(scale)
    level3 = run_level3(scale)
    ir_data, marker_details = _ir_detail(scale)

    result = {
        "level12": level12,
        "level3": level3,
        "snapshot": _snapshot_comparison(),
        "ir_detail": ir_data,
        "marker_details": marker_details,
    }
    try:
        _save_run(level12, level3, ir_data, scale)
    except Exception as e:  # noqa: BLE001
        print(f"[all_verify DB 저장 실패] {e}")
    save_cache(scale, result)
    return result
