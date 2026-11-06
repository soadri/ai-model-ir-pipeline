"""
tolerance_policy.py — RNGD NPU 연산 정합성 오차 허용 기준 정책

설계 근거:
  bf16 (bfloat16): 1 sign + 8 exponent + 7 mantissa
  유효 자릿수 ~3자리, 상대 오차 ~0.8~1.2%

  bf16 ULP (Unit in the Last Place):
    값 1.0 부근 → ULP ≈ 0.0078
    값 10.0 부근 → ULP ≈ 0.125
    값 100.0 부근 → ULP ≈ 1.0

  오차 누적 모델:
    단일 bf16 변환: max 0.5 ULP
    N번 누적 연산(행렬곱 등): ~sqrt(N) * ULP
    체인 연결 M단계: 각 단계 오차가 독립이면 ~sqrt(M) 배 증가

정책 원칙:
  1. 절대 오차와 상대 오차 중 큰 쪽을 기준으로 판단 (max 방식)
  2. 연산 카테고리별 기준을 설정하고, 체인 깊이에 따라 보정
  3. 프로파일(dim 크기)에 따라 스케일 보정 가능
"""


# ══════════════════════════════════════════════════════════════════
# 연산 카테고리별 허용 기준
# ══════════════════════════════════════════════════════════════════

TOLERANCE_POLICY = {
    # ── 카테고리 A: Elementwise (단항/이항) ──
    # bf16 변환 1~2회, 오차 최소
    # 근거: 값 범위 ~1.0, ULP ≈ 0.008, 변환 1회 → max 0.004
    "elementwise": {
        "abs_tol": 1e-3,       # 절대 오차 허용
        "rel_tol": 0.01,       # 상대 오차 1%
        "desc": "단일 bf16 변환 오차 범위. RMSNorm 내부 op, SiLU 등.",
    },

    # ── 카테고리 B: Contraction (행렬곱) ──
    # K차원 누적 덧셈 → sqrt(K) * ULP
    # 근거: K=16일 때 sqrt(16)=4, ULP≈0.008 → ~0.032
    #       K=8일 때 sqrt(8)≈2.8, ULP≈0.125(값 10 부근) → ~0.35
    "contraction": {
        "abs_tol": 0.5,        # 절대 오차 허용
        "rel_tol": 0.05,       # 상대 오차 5%
        "desc": "행렬곱 누적 오차. K차원 크기에 비례하여 증가.",
    },

    # ── 카테고리 C: Reduce ──
    # 트리 구조 4단계 → bf16 변환 4회
    # 근거: 각 단계 ULP 오차 누적, 하지만 f32로 실행하면 정확
    "reduce": {
        "abs_tol": 1e-3,
        "rel_tol": 0.01,
        "desc": "reduce_sum/max. 트리 구조 우회 시 f32 정밀도 유지.",
    },

    # ── 카테고리 D: Layout ──
    # 데이터 이동만, 계산 없음 → 오차 0
    "layout": {
        "abs_tol": 0.0,
        "rel_tol": 0.0,
        "desc": "transpose, broadcast. 오차 없어야 함.",
    },
}


# ══════════════════════════════════════════════════════════════════
# 모듈별 허용 기준 (체인 깊이 반영)
# ══════════════════════════════════════════════════════════════════

MODULE_TOLERANCE = {
    # RMSNorm: elementwise 6단계 체인, 값 범위 ~1.0
    # 근거: pow2→reduce→div→add→rsqrt→mul, 각 단계 ~1e-4 오차
    #       체인 6단계 → sqrt(6) ≈ 2.4배 → ~2.4e-4
    "RMSNORM": {
        "abs_tol": 1e-3,
        "rel_tol": 0.01,
        "chain_depth": 6,
        "desc": "Elementwise 6단계. bf16 변환 최소.",
    },

    # FFN: gemm 3회 + elementwise 3회, 값 범위 ~10
    # 근거: gate(gemm) → SiLU → up(gemm) → mul → down(gemm)
    #       gemm 3회 각 ~0.03, 체인 → ~sqrt(3)*0.03 ≈ 0.05
    #       실측: 0.00093 (f32 경로가 많아서 이론보다 양호)
    "FFN": {
        "abs_tol": 0.01,
        "rel_tol": 0.05,
        "chain_depth": 6,
        "desc": "Gemm 3회 + Elementwise. SiLU는 f32로 실행.",
    },

    # Attention: gemm 4회 + 패딩, 값 범위 ~100
    # 근거: QKV(gemm 3) → scores(gemm) → AV(gemm) → O(gemm)
    #       gemm 4~6회, 각 단계 bf16 변환 + 값 범위 확장
    #       실측: 0.099 (bf16 누적 + 패딩 구조 영향)
    "ATTENTION": {
        "abs_tol": 0.5,
        "rel_tol": 0.05,
        "chain_depth": 10,
        "desc": "Gemm 4+회 체인. bf16 누적 오차 + 패딩 구조.",
    },

    # Full Block: 전체 Decoder Layer
    # 근거: RMSNorm + Attention + residual + RMSNorm + FFN + residual
    #       가장 깊은 체인, 모든 오차 누적
    #       실측: 0.1172
    "FULL_BLOCK": {
        "abs_tol": 1.0,
        "rel_tol": 0.05,
        "chain_depth": 20,
        "desc": "전체 Decoder Layer 체인. 모든 모듈 오차 누적.",
    },
}


# ══════════════════════════════════════════════════════════════════
# 스케일 보정 (프로파일별)
# ══════════════════════════════════════════════════════════════════

SCALE_FACTOR = {
    # dim이 커지면 행렬곱의 K차원이 커져서 오차 증가
    # 보정: sqrt(dim / base_dim)
    "sim_small": {"dim": 16, "factor": 1.0},       # 기준
    "sim_mid":   {"dim": 64, "factor": 2.0},       # sqrt(64/16) = 2
    "llama_8b":  {"dim": 4096, "factor": 16.0},    # sqrt(4096/16) = 16
    "v06_scale": {"dim": 512, "factor": 5.7},      # sqrt(512/16) ≈ 5.7, 0.6 커널 스케일
}


def get_module_tolerance(module_name, profile="sim_small"):
    """모듈별 + 프로파일별 허용 오차 반환"""
    mod = MODULE_TOLERANCE.get(module_name, MODULE_TOLERANCE["FULL_BLOCK"])
    scale = SCALE_FACTOR.get(profile, SCALE_FACTOR["sim_small"])

    return {
        "abs_tol": mod["abs_tol"] * scale["factor"],
        "rel_tol": mod["rel_tol"],  # 상대 오차는 스케일 무관
        "desc": mod["desc"],
        "profile": profile,
        "scale_factor": scale["factor"],
    }


def check_tolerance(module_name, max_diff, ref_max_val=1.0, profile="sim_small"):
    """오차 기준 판정

    Args:
        module_name: RMSNORM, FFN, ATTENTION, FULL_BLOCK
        max_diff: 측정된 최대 절대 오차
        ref_max_val: PyTorch 기준 출력의 최대 절대값
        profile: 검증 프로파일

    Returns:
        (passed, reason)
    """
    tol = get_module_tolerance(module_name, profile)
    abs_ok = max_diff <= tol["abs_tol"]
    rel_diff = max_diff / max(abs(ref_max_val), 1e-10)
    rel_ok = rel_diff <= tol["rel_tol"]

    passed = abs_ok or rel_ok  # 둘 중 하나만 만족하면 PASS

    if passed:
        reason = "abs_diff={:.6f} <= {:.4f} (abs_tol)".format(max_diff, tol["abs_tol"])
        if not abs_ok:
            reason = "rel_diff={:.4f}% <= {:.1f}% (rel_tol)".format(rel_diff*100, tol["rel_tol"]*100)
    else:
        reason = "FAIL: abs_diff={:.6f} > {:.4f} AND rel_diff={:.4f}% > {:.1f}%".format(
            max_diff, tol["abs_tol"], rel_diff*100, tol["rel_tol"]*100)

    return passed, reason


if __name__ == "__main__":
    print("=== 오차 허용 기준 정책 ===\n")

    print("연산 카테고리별:")
    for cat, tol in TOLERANCE_POLICY.items():
        print(f"  {cat:15s} abs<={tol['abs_tol']}, rel<={tol['rel_tol']*100}% — {tol['desc']}")

    print("\n모듈별 (sim_small 기준):")
    for mod, tol in MODULE_TOLERANCE.items():
        print(f"  {mod:15s} abs<={tol['abs_tol']}, rel<={tol['rel_tol']*100}% — {tol['desc']}")

    print("\n스케일 보정:")
    for name, s in SCALE_FACTOR.items():
        print(f"  {name:12s} dim={s['dim']:>5d}, factor={s['factor']}x")

    # 실측값 판정
    print("\n=== 실측값 판정 (sim_small) ===\n")
    tests = [
        ("RMSNORM",    1.4e-05, 1.0),
        ("FFN",        0.00093, 10.0),
        ("ATTENTION",  0.099,   100.0),
        ("FULL_BLOCK", 0.1172,  100.0),
    ]
    for mod, diff, ref_max in tests:
        passed, reason = check_tolerance(mod, diff, ref_max)
        icon = "✅" if passed else "❌"
        print(f"  {icon} {mod:15s} max_diff={diff:<12.6f} {reason}")
