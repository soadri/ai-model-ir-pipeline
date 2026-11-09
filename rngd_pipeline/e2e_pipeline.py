"""
RNGD lowering 파일럿 — v6: Family A(elementwise) + Family B(contraction) 통합 자동화

deploy_and_test.py는 변경 없이 그대로 재사용한다 (e2e_{op}_kernel.rs 등
파일명 규칙만 지키면 어떤 연산이든 배치+테스트가 되도록 이미 설계되어 있음).
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import torch
import torch_mlir
from torch_mlir.ir import (
    Context, Module, Operation, InsertionPoint, StringAttr, RankedTensorType, FloatAttr,
)


def _list_op_names(op) -> list:
    """op을 루트로 모든 하위 op 이름을 재귀 수집 (builtin.module 자신은 제외)."""
    names = []

    def walk(o, is_root=False):
        if not is_root:
            names.append(str(o.name))
        for region in o.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner)

    walk(op, is_root=True)
    return names


def _removed_added(ops_before: list, ops_after: list):
    """before/after 사이에서 사라진 op 이름 집합과 새로 생긴 op 이름 집합을 구한다."""
    before_c, after_c = Counter(ops_before), Counter(ops_after)
    removed = before_c - after_c
    added = after_c - before_c
    return set(removed.keys()), set(added.keys())


def _diff_summary(ops_before: list, ops_after: list) -> str:
    """before/after op 목록을 비교해 '무엇이 사라지고 무엇이 생겼는지' 요약 텍스트 생성."""
    before_c, after_c = Counter(ops_before), Counter(ops_after)
    removed = before_c - after_c
    added = after_c - before_c

    def fmt(counter):
        if not counter:
            return "(없음)"
        return ", ".join(f"{name}×{cnt}" if cnt > 1 else name for name, cnt in counter.items())

    return (
        f"op 개수: {len(ops_before)}개 → {len(ops_after)}개\n"
        f"제거된 op: {fmt(removed)}\n"
        f"추가된 op: {fmt(added)}"
    )


HL_MARK = "@@HL@@"
HL_SEP = "@@SEP@@"


def _mark_ir_annotated(ir_text: str, annotations: list) -> str:
    """annotations: (matcher_substring, label) 목록. 각 줄에서 매칭되는 라벨을 찾아
    HL_MARK<라벨>HL_SEP<원본줄> 형태로 감싼다. 프론트엔드가 이 마커를 보고
    하이라이트 + 라벨 텍스트를 함께 표시한다."""
    if not annotations:
        return ir_text
    lines = ir_text.splitlines()
    out = []
    for line in lines:
        labels = [label for matcher, label in annotations if matcher and matcher in line]
        labels = list(dict.fromkeys(labels))  # 중복 제거, 순서 유지
        if labels:
            out.append(f"{HL_MARK}{' / '.join(labels)}{HL_SEP}{line}")
        else:
            out.append(line)
    return "\n".join(out)


# =====================================================================
# Family A: elementwise (linalg.generic 기반)
# =====================================================================

ARITH_BINARY_OPS = {
    "arith.addf": "add",
    "arith.subf": "sub",
    "arith.mulf": "mul",
    "arith.divf": "div",
}

RNGD_OP_TO_FP_VARIANT = {
    "add": ("FpBinaryOp::AddF", True),
    "sub": ("FpBinaryOp::SubF", True),
    "mul": ("FpBinaryOp::MulF(FpMulAlu::Mul0)", True),
    "div": ("FpBinaryOp::DivF", True),  # 빌드+테스트 통과로 confirmed 승격
}

# 단항(unary) elementwise 연산. math.rsqrt는 하드웨어에 Rsqrt가 따로 없어서
# Sqrt(FpUnaryOp) 다음 1.0/x(vector_fp_div 스칼라 상수)로 조합해서 구현한다.
MATH_UNARY_OPS = {
    "math.rsqrt": "rsqrt",
    "math.sqrt":  "sqrt",
    "math.exp":   "exp",
    "math.log":   "log",
    "math.tanh":  "tanh",
    "math.sin":   "sin",
    "math.cos":   "cos",
    "math.erf":   "erf",
}

# =====================================================================
# Family B: contraction (named op 기반)
# =====================================================================

NAMED_CONTRACTION_OPS = {
    "linalg.batch_matmul": "batch_gemm",   # confirmed: torch.bmm 실측
    "linalg.matmul": "gemm",                # confirmed: torch.matmul(2D) 실측
    "linalg.matvec": "gemv",                # confirmed: torch.mv 실측
    "linalg.dot": "dot_product",            # confirmed: torch.matmul(1D,1D) 실측 (torch.dot 자체는 미지원)
}


@dataclass
class RngdSpec:
    family: Literal["elementwise", "contraction", "reduce", "layout"]
    rngd_op: str
    shape: list = field(default_factory=list)      # elementwise: 출력 shape
    batch: Optional[int] = None                     # contraction: V
    m: Optional[int] = None
    k: Optional[int] = None
    n: Optional[int] = None
    reduce_axes: Optional[list] = None              # reduce: reduction 차원 인덱스


def rewrite_to_rngd(module: Module):
    """Family A/B 통합 rewrite. 모듈 내 매칭되는 모든 op을 순회하며 변환.
    반환: (list[RngdSpec], annotations_before, annotations_after)
    - specs 리스트가 비어 있으면 변환 대상이 없었음을 의미
    - annotations_* 는 모든 변환의 어노테이션을 누적한 리스트
    - 하위 호환: 단일 op 테스트 시 specs[0]으로 접근 가능"""
    results: dict = {"specs": [], "ann_before": [], "ann_after": []}
    # check_* 함수들이 한 번의 매치에서 사용하는 임시 버퍼.
    # 매치 성공 시 results에 flush하고 초기화한다.
    _cur: dict = {"spec": None, "ann_before": [], "ann_after": []}

    def check_elementwise(op, preceding) -> bool:
        if str(op.name) != "linalg.generic":
            return False
        try:
            iter_types = op.attributes["iterator_types"]
            all_parallel = all("parallel" in str(t) for t in iter_types)
        except KeyError:
            all_parallel = False
        if not all_parallel:
            return False

        num_ins = len(op.operands) - len(op.results)
        body_block = op.regions[0].blocks[0]
        body_op_names = [str(inner.name) for inner in body_block.operations]
        binary_ops_present = [n for n in body_op_names if n in ARITH_BINARY_OPS]
        unary_ops_present = [n for n in body_op_names if n in MATH_UNARY_OPS]
        powf_ops_present = [n for n in body_op_names if n == "math.powf"]
        # body 안에서 무시해도 되는 보조 연산 (상수 로드, 타입 변환)
        ARITH_PASSTHROUGH = {"arith.constant", "arith.truncf", "arith.extf", "arith.sitofp"}
        other_ops = [
            n for n in body_op_names
            if n.startswith(("arith.", "math."))
               and n not in ARITH_BINARY_OPS and n not in MATH_UNARY_OPS
               and n != "math.powf" and n not in ARITH_PASSTHROUGH
        ]
        # --- 단항(unary) elementwise: sigmoid (negf+exp+addf+divf 복합 패턴) ---
        SIGMOID_BODY = {"arith.negf", "math.exp", "arith.addf", "arith.divf", "linalg.yield"}
        body_op_set = set(body_op_names)
        if num_ins == 1 and body_op_set == SIGMOID_BODY:
            lhs = op.operands[0]
            result_type = op.results[0].type
            shape = list(RankedTensorType(result_type).shape)
            rngd_op_name = "sigmoid"

            _cur["ann_before"] = [
                ("arith.negf", "SiLU sigmoid 블록 감지 — negf+exp+addf+divf → rngd.elementwise(sigmoid)"),
                ("linalg.generic", "이 블록 전체가 rngd.elementwise(sigmoid)로 대체됨"),
            ]

            with InsertionPoint(op):
                new_op = Operation.create(
                    "rngd.elementwise",
                    results=[result_type],
                    operands=[lhs],
                    attributes={"op": StringAttr.get(rngd_op_name)},
                    loc=op.location,
                )
            op.results[0].replace_all_uses_with(new_op.results[0])
            op.erase()

            _cur["ann_after"] = [
                ("rngd.elementwise", 'sigmoid: FpUnaryOp::Sigmoid 단독 구현'),
            ]
            _cur["spec"] = RngdSpec(family="elementwise", rngd_op=rngd_op_name, shape=shape)
            return True

        if other_ops:
            return False

        # --- 이항(binary) elementwise: add/sub/mul/div ---
        if num_ins == 2 and len(binary_ops_present) == 1 and not unary_ops_present:
            arith_op_name = binary_ops_present[0]
            rngd_op_name = ARITH_BINARY_OPS[arith_op_name]
            lhs, rhs = op.operands[0], op.operands[1]
            result_type = op.results[0].type
            shape = list(RankedTensorType(result_type).shape)

            _cur["ann_before"] = [
                (arith_op_name, f'연산 종류 결정 → rngd.elementwise attribute op="{rngd_op_name}"'),
                ("linalg.generic", "이 블록 전체가 rngd.elementwise 하나로 대체됨"),
                ("tensor.empty", "출력 버퍼 초기화 — 대체 후 불필요해져 제거됨"),
                ("linalg.yield", "generic 내부 결과 반환 — 대체 후 제거됨"),
            ]

            with InsertionPoint(op):
                new_op = Operation.create(
                    "rngd.elementwise",
                    results=[result_type],
                    operands=[lhs, rhs],
                    attributes={"op": StringAttr.get(rngd_op_name)},
                    loc=op.location,
                )
            op.results[0].replace_all_uses_with(new_op.results[0])
            op.erase()

            _cur["ann_after"] = [
                ("rngd.elementwise", f'실제 연산자 — attribute op="{rngd_op_name}"에 연산 종류가 담김'),
            ]
            _cur["spec"] = RngdSpec(family="elementwise", rngd_op=rngd_op_name, shape=shape)
            return True

        # --- broadcast 이항(binary) elementwise: num_ins=1, outs init가 두 번째 피연산자 ---
        # torch_mlir이 scalar broadcast를 linalg.fill(상수) + outs로 표현하는 패턴.
        # 예: RMSNorm의 mean_sq + eps → generic ins(%mean_sq) outs(%eps_filled) { addf }
        if num_ins == 1 and len(binary_ops_present) == 1 and not unary_ops_present:
            arith_op_name = binary_ops_present[0]
            rngd_op_name = ARITH_BINARY_OPS[arith_op_name]
            lhs = op.operands[0]            # ins[0] — 실제 입력
            rhs = op.operands[1]            # outs[0] init — pre-filled 상수/텐서
            result_type = op.results[0].type
            shape = list(RankedTensorType(result_type).shape)

            _cur["ann_before"] = [
                (arith_op_name, f'broadcast 연산 감지 → rngd.elementwise op="{rngd_op_name}" (outs init가 두 번째 피연산자)'),
                ("linalg.generic", "broadcast binary — ins(1) + outs(1) 구조가 rngd.elementwise로 대체됨"),
            ]

            with InsertionPoint(op):
                new_op = Operation.create(
                    "rngd.elementwise",
                    results=[result_type],
                    operands=[lhs, rhs],
                    attributes={"op": StringAttr.get(rngd_op_name)},
                    loc=op.location,
                )
            op.results[0].replace_all_uses_with(new_op.results[0])
            op.erase()

            _cur["ann_after"] = [
                ("rngd.elementwise", f'broadcast binary — op="{rngd_op_name}" (outs init 흡수)'),
            ]
            _cur["spec"] = RngdSpec(family="elementwise", rngd_op=rngd_op_name, shape=shape)
            return True

        # --- 단항(unary) elementwise: rsqrt 등 ---
        if num_ins == 1 and len(unary_ops_present) == 1 and not binary_ops_present:
            math_op_name = unary_ops_present[0]
            rngd_op_name = MATH_UNARY_OPS[math_op_name]
            lhs = op.operands[0]
            result_type = op.results[0].type
            shape = list(RankedTensorType(result_type).shape)

            _cur["ann_before"] = [
                (math_op_name, f'연산 종류 결정 → rngd.elementwise attribute op="{rngd_op_name}"'),
                ("linalg.generic", "이 블록 전체가 rngd.elementwise 하나로 대체됨"),
                ("tensor.empty", "출력 버퍼 초기화 — 대체 후 불필요해져 제거됨"),
                ("linalg.yield", "generic 내부 결과 반환 — 대체 후 제거됨"),
            ]

            with InsertionPoint(op):
                new_op = Operation.create(
                    "rngd.elementwise",
                    results=[result_type],
                    operands=[lhs],
                    attributes={"op": StringAttr.get(rngd_op_name)},
                    loc=op.location,
                )
            op.results[0].replace_all_uses_with(new_op.results[0])
            op.erase()

            _cur["ann_after"] = [
                ("rngd.elementwise", f'실제 연산자 — attribute op="{rngd_op_name}"에 연산 종류가 담김 (하드웨어엔 Sqrt만 있어 1.0/sqrt(x)로 조합)'),
            ]
            _cur["spec"] = RngdSpec(family="elementwise", rngd_op=rngd_op_name, shape=shape)
            return True

        # --- 단항(unary) elementwise: pow(x, 2.0) -> pow2 (지수가 정확히 2.0인지 실제 값 검증) ---
        if num_ins == 1 and len(powf_ops_present) == 1 and not binary_ops_present and not unary_ops_present:
            powf_op = next(o for o in body_block.operations if str(o.name) == "math.powf")
            exponent_operand = powf_op.operands[1]
            exponent_def = exponent_operand.owner
            exp_val = None
            if exponent_def is not None and hasattr(exponent_def, "attributes"):
                try:
                    exp_val = FloatAttr(exponent_def.attributes["value"]).value
                except (KeyError, ValueError, TypeError):
                    exp_val = None

            if exp_val != 2.0:
                return False

            rngd_op_name = "pow2"
            lhs = op.operands[0]
            result_type = op.results[0].type
            shape = list(RankedTensorType(result_type).shape)

            _cur["ann_before"] = [
                ("math.powf", f'연산 종류 결정 (지수={exp_val}) -> rngd.elementwise attribute op="{rngd_op_name}"'),
                ("linalg.generic", "이 블록 전체가 rngd.elementwise 하나로 대체됨"),
                ("tensor.empty", "출력 버퍼 초기화 — 대체 후 불필요해져 제거됨"),
                ("linalg.yield", "generic 내부 결과 반환 — 대체 후 제거됨"),
            ]

            with InsertionPoint(op):
                new_op = Operation.create(
                    "rngd.elementwise",
                    results=[result_type],
                    operands=[lhs],
                    attributes={"op": StringAttr.get(rngd_op_name)},
                    loc=op.location,
                )
            op.results[0].replace_all_uses_with(new_op.results[0])
            op.erase()

            _cur["ann_after"] = [
                ("rngd.elementwise", f'실제 연산자 — attribute op="{rngd_op_name}"에 연산 종류가 담김 (하드웨어엔 Pow가 없어 MulF(x,x)로 조합)'),
            ]
            _cur["spec"] = RngdSpec(family="elementwise", rngd_op=rngd_op_name, shape=shape)
            return True

        return False

    def _is_identity_generic(op) -> bool:
        """linalg.generic이 단순 identity (passthrough)인지 확인.
        body가 linalg.yield 하나뿐이고, yield 값이 첫 번째 block arg (입력)이면 identity."""
        try:
            if str(op.name) != "linalg.generic":
                return False
            iter_types = op.attributes["iterator_types"]
            if not all("parallel" in str(t) for t in iter_types):
                return False
            body = op.regions[0].blocks[0]
            body_ops = list(body.operations)
            if len(body_ops) != 1 or str(body_ops[0].name) != "linalg.yield":
                return False
            yield_operand = body_ops[0].operands[0]
            block_args = list(body.arguments)
            return len(block_args) >= 1 and yield_operand == block_args[0]
        except (RuntimeError, ValueError, KeyError):
            return False

    def _identity_layout(op):
        """identity generic이 실제로 무엇인지 indexing_maps로 판별한다.

        shape만 보면 축 크기가 같을 때(예: [1,8,8,8]) 전치와 확장을 구분할 수 없으므로,
        입력·출력 indexing map의 관계로 판정한다. 출력 축 j는 입력 축 perm[j]에서 온다.
        반환: ("transpose", perm) / ("broadcast", 늘어난 출력 축 목록) / ("copy", []) / (None, None)
        """
        try:
            maps = [str(m) for m in op.attributes["indexing_maps"]]
            parsed = []
            for m in maps:
                mm = re.search(r"\(([^)]*)\)\s*->\s*\(([^)]*)\)", m)
                parsed.append(([d.strip() for d in mm.group(1).split(",") if d.strip()],
                               [r.strip() for r in mm.group(2).split(",") if r.strip()]))
            dims, rin = parsed[0]
            _, rout = parsed[-1]
            pure = lambda r: len(r) == len(dims) and all(x in dims for x in r) and len(set(r)) == len(r)
            if not pure(rout):
                return (None, None)
            if not pure(rin):
                return ("broadcast", [j for j, d in enumerate(rout) if d not in rin])
            perm = [rin.index(d) for d in rout]
            if perm == list(range(len(perm))):
                return ("copy", [])
            return ("transpose", perm)
        except (KeyError, AttributeError, RuntimeError, ValueError):
            return (None, None)

    def _is_transpose_generic(op) -> bool:
        """linalg.generic이 마지막 두 축만 바꾸는 전치인지 확인 (indexing_maps 기준).
        torch_mlir은 linalg.transpose 대신 indexing_maps가 다른 identity generic으로 전치를 표현한다."""
        if not _is_identity_generic(op):
            return False
        kind, perm = _identity_layout(op)
        if kind != "transpose" or len(perm) < 2:
            return False
        n = len(perm)
        return perm == list(range(n - 2)) + [n - 1, n - 2]

    def _owner_op(value):
        """value의 producer Operation을 반환. block argument면 None."""
        owner = value.owner
        if owner is None or not hasattr(owner, "name"):
            return None  # Block 객체이거나 없음
        return owner

    def _try_absorb_transpose_chain(rhs_value):
        """rhs에서 역추적하여 identity generic 체인을 따라가고, transpose를 찾으면 흡수.
        Llama 패턴: weight → transpose_generic → expand_generic → batch_matmul
        최대 5단계까지 identity generic을 따라감.
        성공 시: (original_weight_value, [흡수할 op 리스트]) 반환
        실패 시: None 반환"""
        try:
            absorbed_ops = []
            current = rhs_value

            for _ in range(5):  # 무한 루프 방지
                producer = _owner_op(current)
                if producer is None:
                    break

                # transpose generic 발견 → 성공, 체인 종료
                if _is_transpose_generic(producer):
                    absorbed_ops.append(producer)
                    original_weight = producer.operands[0]
                    return (original_weight, absorbed_ops)

                # linalg.transpose 발견 → fallback 성공
                if str(producer.name) == "linalg.transpose":
                    absorbed_ops.append(producer)
                    original_weight = producer.operands[0]
                    return (original_weight, absorbed_ops)

                # 단순 복사인 identity generic만 흡수하고 계속 따라감
                # (확장이나 다른 축의 전치를 흡수하면 그 정보가 IR에서 사라지므로 체인을 끊는다)
                if _is_identity_generic(producer) and _identity_layout(producer)[0] == "copy":
                    absorbed_ops.append(producer)
                    current = producer.operands[0]  # ins[0]를 따라 올라감
                    continue

                # identity도 transpose도 아닌 op → 체인 끊김
                break

            return None
        except (RuntimeError, ValueError, AttributeError):
            return None

    def _dce_op_and_allocs(op_to_erase):
        """op을 삭제하고, 그 operand 중 uses가 0인 producer(tensor.empty 등)도 정리."""
        try:
            operands = list(op_to_erase.operands)
            op_to_erase.erase()
            for operand in operands:
                producer = _owner_op(operand)
                if producer is not None and hasattr(producer, "results"):
                    try:
                        if len(list(producer.results[0].uses)) == 0:
                            producer.erase()
                    except (RuntimeError, ValueError):
                        pass
        except (RuntimeError, ValueError):
            pass


    # reduction 연산 종류 매핑 (body 내 연산 → reduce op 이름)
    REDUCE_BODY_OPS = {
        "arith.addf": "sum",        # RMSNorm mean, Softmax 분모
        "arith.maximumf": "max",    # Softmax max (수치 안정성)
        "arith.maxf": "max",        # 일부 torch_mlir 버전
    }

    def check_reduce(op) -> bool:
        """linalg.generic에서 reduction iterator_type을 가진 reduce 연산을 감지."""
        if str(op.name) != "linalg.generic":
            return False
        try:
            iter_types = op.attributes["iterator_types"]
            iter_strs = [str(t) for t in iter_types]
        except KeyError:
            return False

        reduction_axes = [i for i, t in enumerate(iter_strs) if "reduction" in t]
        if not reduction_axes:
            return False

        body_block = op.regions[0].blocks[0]
        body_op_names = [str(inner.name) for inner in body_block.operations]
        reduce_op_found = None
        for body_op_name in body_op_names:
            if body_op_name in REDUCE_BODY_OPS:
                reduce_op_found = body_op_name
                break

        if reduce_op_found is None:
            return False

        rngd_op_name = REDUCE_BODY_OPS[reduce_op_found]
        result_type = op.results[0].type
        out_shape = list(RankedTensorType(result_type).shape)
        inp = op.operands[0]

        _cur["ann_before"] = [
            (reduce_op_found, f'reduce body 감지 → rngd.reduce(op="{rngd_op_name}", axes={reduction_axes})'),
            ("linalg.generic", f"reduction 축 {reduction_axes} → rngd.reduce로 대체"),
        ]

        with InsertionPoint(op):
            new_op = Operation.create(
                "rngd.reduce",
                results=[result_type],
                operands=[inp],
                attributes={
                    "op": StringAttr.get(rngd_op_name, context=op.context),
                },
                loc=op.location,
            )
        op.results[0].replace_all_uses_with(new_op.results[0])
        op.erase()

        _cur["ann_after"] = [
            ("rngd.reduce", f'reduce_{rngd_op_name}: K-Cloud 트리 우회(elementwise add x4) 또는 vector_inter_slice_reduce로 실행'),
        ]
        _cur["spec"] = RngdSpec(
            family="reduce", rngd_op=f"reduce_{rngd_op_name}",
            shape=out_shape, reduce_axes=reduction_axes,
        )
        return True

    def check_contraction(op) -> bool:
        if str(op.name) not in NAMED_CONTRACTION_OPS:
            return False
        rngd_op_name = NAMED_CONTRACTION_OPS[str(op.name)]
        original_op_name = str(op.name)

        lhs, rhs, outs_init = op.operands[0], op.operands[1], op.operands[2]
        result_type = op.results[0].type
        shape = list(RankedTensorType(result_type).shape)       # [V,M,N] 또는 [M,N]
        lhs_shape = list(RankedTensorType(lhs.type).shape)        # [V,M,K] 또는 [M,K]

        # ── 3-op chain 감지: [identity_generic →] transpose → batch_matmul ──
        transpose_rhs = False
        absorbed_ops = []
        chain_info = _try_absorb_transpose_chain(rhs)
        if chain_info is not None:
            original_weight, absorbed_ops = chain_info
            transpose_rhs = True
            rhs = original_weight  # 원본 가중치 (전치 전) 사용

        # ── 어노테이션 (before) ──
        ann_items = [
            (original_op_name, f"이 연산 전체가 rngd.{rngd_op_name} 하나로 대체됨"),
            ("linalg.fill", "출력 0-초기화 — 대체 후 흡수되어 사라짐 (DCE로 제거)"),
        ]
        if transpose_rhs:
            ann_items.append(("linalg.generic", "transpose generic 감지 — identity body + shape 전치 → rngd 속성(transpose_rhs)으로 흡수됨"))
        _cur["ann_before"] = ann_items

        # ── op 생성 ──
        attrs = {}
        if transpose_rhs:
            attrs["transpose_rhs"] = StringAttr.get("true", context=op.context)

        with InsertionPoint(op):
            new_op = Operation.create(
                f"rngd.{rngd_op_name}",
                results=[result_type],
                operands=[lhs, rhs],
                attributes=attrs,
                loc=op.location,
            )
        op.results[0].replace_all_uses_with(new_op.results[0])
        op.erase()

        # ── DCE: linalg.fill + tensor.empty + arith.constant ──
        fill_op = outs_init.owner
        if fill_op is not None and str(fill_op.name) == "linalg.fill":
            if len(list(fill_op.results[0].uses)) == 0:
                _dce_op_and_allocs(fill_op)

        # ── DCE: 흡수된 transpose/identity chain ──
        for absorbed in absorbed_ops:
            try:
                if len(list(absorbed.results[0].uses)) == 0:
                    _dce_op_and_allocs(absorbed)
            except (RuntimeError, ValueError):
                pass

        # ── shape 추출 (batch_matmul 기준, transpose 흡수 전 shape 사용) ──
        if len(shape) == 3:
            batch, m, n = shape
            k = lhs_shape[2]
        elif len(shape) == 2:
            batch = None
            m, n = shape
            k = lhs_shape[1]
        else:
            batch = None
            m, n = 1, 1
            k = lhs_shape[0]

        chain_label = "(transpose_rhs=true) " if transpose_rhs else ""
        absorbed_count = len(absorbed_ops)
        _cur["ann_after"] = [
            (f"rngd.{rngd_op_name}", f"{chain_label}Contraction Engine 실행 — {absorbed_count + 1}개 op 흡수" if absorbed_count else "Contraction Engine(TRF + contract_outer/packet/time/lane)이 실제 계산 수행"),
        ]
        _cur["spec"] = RngdSpec(family="contraction", rngd_op=rngd_op_name, batch=batch, m=m, n=n, k=k)
        return True

    def _flush_cur():
        """현재 매치 결과를 results에 누적하고 _cur를 초기화."""
        if _cur["spec"] is not None:
            results["specs"].append(_cur["spec"])
            results["ann_before"].extend(_cur["ann_before"])
            results["ann_after"].extend(_cur["ann_after"])
        _cur["spec"] = None
        _cur["ann_before"] = []
        _cur["ann_after"] = []

    def _is_live(op) -> bool:
        """DCE 등으로 이미 삭제된 op인지 안전하게 확인."""
        try:
            _ = str(op.name)
            return True
        except (RuntimeError, ValueError):
            return False

    def walk_block(block, preceding=None):
        preceding = preceding or []
        for op in list(block.operations):
            if not _is_live(op):
                continue
            if check_elementwise(op, preceding) or check_reduce(op) or check_contraction(op):
                _flush_cur()
                continue  # 변환 완료된 op — 하위 region 순회 불필요
            for region in op.regions:
                for inner_block in region.blocks:
                    walk_block(inner_block)

    walk_block(module.operation.regions[0].blocks[0])

    # ── 2nd pass: 잔류 identity generic → rngd.transpose / rngd.broadcast 변환 ──
    # 1st pass(walk_block)에서 contraction chain 흡수가 끝난 후 실행.
    # 남아있는 identity generic은 standalone 레이아웃 변환이므로 안전하게 변환 가능.
    def convert_remaining_identity(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if not _is_live(inner):
                        continue
                    if str(inner.name) == "linalg.generic" and _is_identity_generic(inner):
                        try:
                            in_shape = list(RankedTensorType(inner.operands[0].type).shape)
                            out_shape = list(RankedTensorType(inner.results[0].type).shape)
                            result_type = inner.results[0].type
                            inp = inner.operands[0]

                            # indexing_maps로 전치·확장·복사 판별 (shape만 보면 축 크기가 같을 때 구분 불가)
                            kind, info = _identity_layout(inner)
                            attrs = {
                                "in_shape": StringAttr.get(str(in_shape), context=inner.context),
                                "out_shape": StringAttr.get(str(out_shape), context=inner.context),
                            }
                            if kind == "transpose":
                                rngd_op, label = "rngd.transpose", "transpose"
                                attrs["perm"] = StringAttr.get(str(info), context=inner.context)
                                detail = f"perm={info}"
                            elif kind in ("broadcast", "copy"):
                                rngd_op, label = "rngd.broadcast", "broadcast"
                                attrs["broadcast_dims"] = StringAttr.get(str(info), context=inner.context)
                                detail = f"broadcast_dims={info}" if info else "복사"
                            else:
                                # map을 해석하지 못하면 변환하지 않고 남겨 미분류로 드러나게 한다
                                continue

                            with InsertionPoint(inner):
                                new_op = Operation.create(
                                    rngd_op,
                                    results=[result_type],
                                    operands=[inp],
                                    attributes=attrs,
                                    loc=inner.location,
                                )
                            inner.results[0].replace_all_uses_with(new_op.results[0])
                            inner.erase()

                            results["specs"].append(RngdSpec(
                                family="layout", rngd_op=label, shape=out_shape,
                            ))
                            results["ann_after"].append(
                                (rngd_op, f"{label}: {in_shape} → {out_shape} ({detail})")
                            )
                        except (RuntimeError, ValueError):
                            pass
                    else:
                        convert_remaining_identity(inner)

    convert_remaining_identity(module.operation)

    return results["specs"], results["ann_before"], results["ann_after"]


# =====================================================================
# 코드 생성 — Family A
# =====================================================================

def _gen_kernel_elementwise(rngd_op: str, axis_size: int):
    fp_variant, confirmed = RNGD_OP_TO_FP_VARIANT[rngd_op]
    status = "confirmed" if confirmed else "EXPERIMENTAL"

    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.elementwise(op="{rngd_op}") — {status}
axes![A = {axis_size}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn pilot_e2e_{rngd_op}_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<f32, Chip, m![A]>,
    rhs: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {{
    let lhs_dm = lhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);
    let rhs_dm = rhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 1 << 12);

    let rhs_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![A % 8]> = ctx
        .sub
        .begin(rhs_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![A % 8 / 8], m![A % 8 % 8]>()
        .to_vrf(0);

    let result = ctx
        .main
        .begin(lhs_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary({fp_variant}, &rhs_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>(1 << 13);

    result.to_hbm(&mut ctx.tdma, 1 << 28)
}}
"""

    host_and_test_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_{rngd_op}_kernel::{{A, pilot_e2e_{rngd_op}_kernel}};

mod reference_data_e2e_{rngd_op};
use reference_data_e2e_{rngd_op}::{{CHECK_N, reference_a, reference_b, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let lhs = HostTensor::<f32, m![A]>::from_buf(reference_a());
    let rhs = HostTensor::<f32, m![A]>::from_buf(reference_b());
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_e2e_{rngd_op}_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Pilot E2E {rngd_op}: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {{
        let mut ctx = Context::acquire();
        let lhs = HostTensor::<f32, m![A]>::from_buf(reference_a());
        let rhs = HostTensor::<f32, m![A]>::from_buf(reference_b());
        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

        let out_hbm = launch(pilot_e2e_{rngd_op}_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {{
            println!("  [{{i}}]: {{}} | {{}}", f32::from(expected[i]), f32::from(actual[i]));
        }}
        for i in 0..CHECK_N {{
            assert!(
                (expected[i] - actual[i]).abs() < 1e-4,
                "mismatch at i={{i}}: pytorch={{}}  rngd_sim={{}}", expected[i], actual[i]
            );
        }}
    }}
}}
"""
    return kernel_rs, host_and_test_rs


def _gen_reference_elementwise(a, b, expected, axis_size):
    check_n = len(a)
    a_full = a + [0.0] * (axis_size - check_n)
    b_full = b + [0.0] * (axis_size - check_n)

    def lit(arr):
        return "vec![" + ", ".join(f"{float(v):.8}_f32" for v in arr) + "]"

    return (
        "// AUTO-GENERATED — expected는 실제 PyTorch forward() 출력값\n"
        f"pub const CHECK_N: usize = {check_n};\n"
        f"pub fn reference_a() -> Vec<f32> {{ {lit(a_full)} }}\n"
        f"pub fn reference_b() -> Vec<f32> {{ {lit(b_full)} }}\n"
        f"pub fn reference_expected() -> Vec<f32> {{ {lit(expected)} }}\n"
    )


# =====================================================================
# 코드 생성 — Family A 단항(unary) elementwise (rsqrt 등)
# =====================================================================

# 하드웨어에 직접 있는 FpUnaryOp 하나로 안 끝나는 경우, 뒤에 이어붙일 체인.
# rsqrt = 1.0 / sqrt(x): Sqrt(단항) 다음 스칼라 상수 1.0으로 나누기.
UNARY_OP_CHAIN = {
    "rsqrt": ".vector_fp_unary(FpUnaryOp::Sqrt)\n        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0_f32)",
    "sqrt":  ".vector_fp_unary(FpUnaryOp::Sqrt)",
    "exp":   ".vector_fp_unary(FpUnaryOp::Exp)",
    "sigmoid": ".vector_fp_unary(FpUnaryOp::Sigmoid)",
    "silu":    ".vector_stash()\n        .vector_fp_unary(FpUnaryOp::Sigmoid)\n        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)",
    "log":     ".vector_fp_unary(FpUnaryOp::Log)",
    "tanh":    ".vector_fp_unary(FpUnaryOp::Tanh)",
    "sin":     ".vector_fp_unary(FpUnaryOp::Sin)",
    "cos":     ".vector_fp_unary(FpUnaryOp::Cos)",
    "erf":     ".vector_fp_unary(FpUnaryOp::Erf)",
}


def _gen_kernel_elementwise_unary(rngd_op: str, axis_size: int):
    chain = UNARY_OP_CHAIN[rngd_op]

    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.elementwise(op="{rngd_op}") — 단항(unary), EXPERIMENTAL
// 하드웨어엔 Rsqrt가 직접 없어서 Sqrt(FpUnaryOp) 다음 1.0/x(vector_fp_div 스칼라)로 조합.
axes![A = {axis_size}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn pilot_e2e_{rngd_op}_kernel(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {{
    let input_dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);

    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        {chain}
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>(1 << 13);

    result.to_hbm(&mut ctx.tdma, 1 << 28)
}}
"""

    host_and_test_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_{rngd_op}_kernel::{{A, pilot_e2e_{rngd_op}_kernel}};

mod reference_data_e2e_{rngd_op};
use reference_data_e2e_{rngd_op}::{{CHECK_N, reference_a, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let input = HostTensor::<f32, m![A]>::from_buf(reference_a());
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;
    let _out_hbm = launch(pilot_e2e_{rngd_op}_kernel, (&mut ctx, &input_hbm)).await;
    println!("Pilot E2E {rngd_op}: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {{
        let mut ctx = Context::acquire();
        let input = HostTensor::<f32, m![A]>::from_buf(reference_a());
        let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;

        let out_hbm = launch(pilot_e2e_{rngd_op}_kernel, (&mut ctx, &input_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {{
            println!("  [{{i}}]: {{}} | {{}}", f32::from(expected[i]), f32::from(actual[i]));
        }}
        for i in 0..CHECK_N {{
            let diff = (expected[i] - actual[i]).abs();
            let tol = (0.02 * expected[i].abs()).max(1e-3);
            assert!(
                diff <= tol,
                "mismatch at i={{i}}: pytorch={{}}  rngd_sim={{}}  diff={{}} > tol={{}}", expected[i], actual[i], diff, tol
            );
        }}
    }}
}}
"""
    return kernel_rs, host_and_test_rs


def _gen_reference_elementwise_unary(a, expected, axis_size):
    check_n = len(a)
    a_full = a + [1.0] * (axis_size - check_n)   # rsqrt(0)=inf 방지용으로 0 대신 1.0 패딩
    expected_full = expected + [1.0] * (axis_size - check_n)

    def lit(arr):
        return "vec![" + ", ".join(f"{float(v):.8}_f32" for v in arr) + "]"

    return (
        "// AUTO-GENERATED — expected는 실제 PyTorch forward() 출력값\n"
        f"pub const CHECK_N: usize = {check_n};\n"
        f"pub fn reference_a() -> Vec<f32> {{ {lit(a_full)} }}\n"
        f"pub fn reference_expected() -> Vec<f32> {{ {lit(expected_full)} }}\n"
    )


def _gen_kernel_elementwise_square(axis_size: int):
    """x^2 = x*x. 하드웨어에 Pow가 없어서, 같은 입력을 두 번(메인 스트림 + VRF로)
    적재해 MulF(x,x)로 구현한다 — 이항 elementwise 커널과 거의 동일한 구조."""
    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.elementwise(op="pow2") — 단항(unary), EXPERIMENTAL
// 하드웨어엔 Pow가 없어서 같은 입력을 두 번 적재해 MulF(x,x)로 조합.
axes![A = {axis_size}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn pilot_e2e_pow2_kernel(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {{
    let input_dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);

    // 같은 입력을 VRF로도 적재 — 자기 자신과 곱하기 위함
    let self_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![A % 8]> = ctx
        .sub
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![A % 8 / 8], m![A % 8 % 8]>()
        .to_vrf(0);

    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>(1 << 13);

    result.to_hbm(&mut ctx.tdma, 1 << 28)
}}
"""

    host_and_test_rs = """\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_pow2_kernel::{A, pilot_e2e_pow2_kernel};

mod reference_data_e2e_pow2;
use reference_data_e2e_pow2::{CHECK_N, reference_a, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<f32, m![A]>::from_buf(reference_a());
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;
    let _out_hbm = launch(pilot_e2e_pow2_kernel, (&mut ctx, &input_hbm)).await;
    println!("Pilot E2E pow2: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {
        let mut ctx = Context::acquire();
        let input = HostTensor::<f32, m![A]>::from_buf(reference_a());
        let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;

        let out_hbm = launch(pilot_e2e_pow2_kernel, (&mut ctx, &input_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {
            println!("  [{}]: {} | {}", i, expected[i], actual[i]);
        }
        for i in 0..CHECK_N {
            let diff = (expected[i] - actual[i]).abs();
            let tol = (0.02 * expected[i].abs()).max(1e-3);
            assert!(
                diff <= tol,
                "mismatch at i={}: pytorch={}  rngd_sim={}  diff={} > tol={}", i, expected[i], actual[i], diff, tol
            );
        }
    }
}
"""
    return kernel_rs, host_and_test_rs


# =====================================================================
# 코드 생성 — Family B (batch_gemm)
# =====================================================================

def _gen_kernel_batch_gemm(v: int, m: int, k: int, n: int, op_name: str = "batch_gemm"):
    assert v <= 256, f"V={v} > 256: 이 템플릿은 V<=256 가정(Slice=m![V # 256])에서만 검증됨"

    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.{op_name} — confirmed (pilot_batch_gemm 구조 재사용)
// 배치 축이 없는 2D matmul(gemm)은 V=1인 batch_gemm으로 취급 (동일 검증 구조 재사용)
axes![V = {v}, M = {m}, K = {k}, N = {n}];

pub type Chip = m![1];
pub type Cluster = m![1];
// V<=256 가정: 하드웨어 제약(Slice ∈ {{64,128,192,256}})에 맞춰 256-slice 공간에 매핑
pub type Slice = m![V # 256];
// contract_lane의 OutPacket::SIZE는 정확히 8이어야 함(실측 확인) -> N<8이면 8-lane 공간에 패딩
pub type Lane = m![N # 8];

#[device(chip = 1)]
pub fn pilot_e2e_{op_name}_kernel(
    ctx: &mut Context,
    a: &HbmTensor<bf16, Chip, m![V, M, K]>,
    b: &HbmTensor<bf16, Chip, m![V, K, N]>,
) -> HbmTensor<bf16, Chip, m![V, M, N]> {{
    let a: DmTensor<bf16, Chip, Cluster, Slice, m![M, K]> = a.to_dm(&mut ctx.tdma, 0);
    let b: DmTensor<bf16, Chip, Cluster, Slice, m![K, N]> = b.to_dm(&mut ctx.tdma, 1 << 12);

    let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![K]> = ctx
        .sub
        .begin(b.view())
        .fetch::<m![N], m![K]>()
        .collect::<m![N, K / 16], m![K % 16]>()
        .to_trf(TrfAddress::Full);

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![M, N]> = ctx
        .main
        .begin(a.view())
        .fetch::<m![M], m![K]>()
        .collect::<m![M, K / 16], m![K % 16]>()
        .contract_outer::<m![M], m![K], _, _>(&b_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![M]>()
        .contract_lane::<m![M], m![N # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![N # 16]>()
        .commit_trim::<m![N]>()
        .commit(0);

    result.to_hbm(&mut ctx.tdma, 2 << 28)
}}
"""

    host_and_test_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_{op_name}_kernel::{{V, M, K, N, pilot_e2e_{op_name}_kernel}};

mod reference_data_e2e_{op_name};
use reference_data_e2e_{op_name}::{{reference_a, reference_b, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let a = HostTensor::<bf16, m![V, M, K]>::from_buf(reference_a());
    let b = HostTensor::<bf16, m![V, K, N]>::from_buf(reference_b());
    let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_e2e_{op_name}_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
    println!("Pilot E2E {op_name}: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    /// expected는 실제 PyTorch 모델의 입력을 bf16으로 양자화한 뒤 그 값으로
    /// (f32 누산 + bf16 반올림) 계산한 값이다 — RNGD가 실제로 낼 수 있는
    /// "현실적인 정답"이며, Rust가 자체 재계산한 값이 아니다.
    #[tokio::test]
    async fn matches_bf16_reference() {{
        let mut ctx = Context::acquire();
        let a = HostTensor::<bf16, m![V, M, K]>::from_buf(reference_a());
        let b = HostTensor::<bf16, m![V, K, N]>::from_buf(reference_b());
        let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
        let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;

        let out_hbm = launch(pilot_e2e_{op_name}_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
        let actual: Vec<bf16> = out_hbm.to_host::<m![V, M, N]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== 값 비교 (앞 8개) ===");
        for i in 0..8.min(actual.len()) {{
            println!("  [{{i}}]: {{:?}} | {{:?}}", expected[i], actual[i]);
        }}

        for (idx, (&e, &av)) in expected.iter().zip(&actual).enumerate() {{
            let diff = (f32::from(av) - f32::from(e)).abs();
            let tol = (0.05 * f32::from(e).abs()).max(1.0);
            assert!(diff <= tol, "{op_name} mismatch at idx={{idx}}: expected {{e:?}}, actual {{av:?}}");
        }}
    }}
}}
"""
    return kernel_rs, host_and_test_rs


def _gen_reference_batch_gemm(a_bf16, b_bf16, expected_bf16):
    def build_const(arr, const_name):
        # 함수 호출(bf16::from_f32)을 원소마다 반복하면 AST 노드가 수만 개가 되어
        # 컴파일이 느려짐. 대신 순수 숫자 리터럴 배열(파싱만 하면 됨) + 단일
        # .map(bf16::from_f32) 호출로 바꿔 호출 표현식은 소스에 1번만 나오게 한다.
        values_str = ", ".join(f"{float(v):.8}_f32" for v in arr)
        return f"const {const_name}: [f32; {len(arr)}] = [{values_str}];\n"

    a_const = build_const(a_bf16, "A_F32_VALUES")
    b_const = build_const(b_bf16, "B_F32_VALUES")
    e_const = build_const(expected_bf16, "EXPECTED_F32_VALUES")

    return (
        "// AUTO-GENERATED — expected는 실제 PyTorch 입력을 bf16 양자화 후 계산한 값\n"
        "use furiosa_opt_std::prelude::*;\n\n"
        f"{a_const}"
        f"{b_const}"
        f"{e_const}"
        "pub fn reference_a() -> Vec<bf16> { A_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
        "pub fn reference_b() -> Vec<bf16> { B_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
        "pub fn reference_expected() -> Vec<bf16> { EXPECTED_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
    )


# =====================================================================
# 코드 생성 — gemv (gemv_kernel.rs 구조 그대로 반영)
# =====================================================================

def _gen_kernel_gemv(i: int, j: int):
    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.gemv — 검증된 gemv_kernel.rs 구조 그대로 사용.
axes![I = {i}, J = {j}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![I];
pub type Time = m![J / 32];
pub type Packet = m![J % 32];
pub type Lane = m![1];

#[device(chip = 1)]
pub fn pilot_e2e_gemv_kernel(
    ctx: &mut Context,
    matrix: &HbmTensor<bf16, Chip, m![I, J]>,
    vector: &HbmTensor<bf16, Chip, m![J]>,
) -> HbmTensor<bf16, Chip, m![I]> {{
    let matrix: DmTensor<bf16, Chip, Cluster, Slice, m![J]> = matrix.to_dm(&mut ctx.tdma, 0);
    let vector: DmTensor<bf16, Chip, Cluster, Slice, m![J]> = vector.to_dm(&mut ctx.tdma, 1 << 12);

    let vector_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![J]> = ctx
        .sub
        .begin(vector.view())
        .fetch::<m![1], m![J]>()
        .collect::<m![J / 16], m![J % 16]>()
        .to_trf(TrfAddress::Full);

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 4]> = ctx
        .main
        .begin(matrix.view())
        .fetch::<m![J / 16], m![J % 16]>()
        .collect::<m![J / 16], m![J % 16]>()
        .contract_outer::<Time, Packet, _, _>(&vector_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>()
        .commit(0);

    result.to_hbm(&mut ctx.tdma, 2 << 28)
}}
"""
    return kernel_rs


def _gen_pilot_gemv(i: int, j: int):
    pilot_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_gemv_kernel::{{pilot_e2e_gemv_kernel, I, J}};

mod reference_data_e2e_gemv;
use reference_data_e2e_gemv::{{CHECK_N, reference_matrix, reference_vector, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let matrix = HostTensor::<bf16, m![I, J]>::from_buf(reference_matrix());
    let vector = HostTensor::<bf16, m![J]>::from_buf(reference_vector());
    let matrix_hbm = matrix.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let vector_hbm = vector.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_e2e_gemv_kernel, (&mut ctx, &matrix_hbm, &vector_hbm)).await;
    println!("Pilot E2E gemv: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {{
        let mut ctx = Context::acquire();
        let matrix = HostTensor::<bf16, m![I, J]>::from_buf(reference_matrix());
        let vector = HostTensor::<bf16, m![J]>::from_buf(reference_vector());
        let matrix_hbm = matrix.to_hbm(&mut ctx.pdma, 0 << 28).await;
        let vector_hbm = vector.to_hbm(&mut ctx.pdma, 1 << 28).await;

        let out_hbm = launch(pilot_e2e_gemv_kernel, (&mut ctx, &matrix_hbm, &vector_hbm)).await;
        let actual: Vec<bf16> = out_hbm.to_host::<m![I]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {{
            println!("  [{{i}}]: {{}} | {{}}", f32::from(expected[i]), f32::from(actual[i]));
        }}
        for i in 0..CHECK_N {{
            let e = f32::from(expected[i]);
            let a = f32::from(actual[i]);
            let tol = (e.abs() * 0.02_f32).max(0.5_f32);
            assert!(
                (e - a).abs() < tol,
                "mismatch at i={{i}}: pytorch={{}}  rngd_sim={{}}", e, a
            );
        }}
    }}
}}
"""
    return pilot_rs


def _gen_reference_gemv(matrix_bf16, vector_bf16, expected_bf16):
    def fmt_f32(t):
        return ", ".join(f"{v:.8f}_f32" for v in t.to(torch.float32).flatten().tolist())

    mat_f32  = fmt_f32(matrix_bf16)
    vec_f32  = fmt_f32(vector_bf16)
    exp_f32  = fmt_f32(expected_bf16)
    check_n  = expected_bf16.numel()

    ref_rs = f"""\
// AUTO-GENERATED — expected는 실제 PyTorch 입력을 bf16 양자화 후 계산한 값
use furiosa_opt_std::prelude::*;

pub const CHECK_N: usize = {check_n};

const MAT_F32: &[f32] = &[{mat_f32}];
const VEC_F32: &[f32] = &[{vec_f32}];
const EXP_F32: &[f32] = &[{exp_f32}];

pub fn reference_matrix() -> Vec<bf16> {{ MAT_F32.iter().map(|&v| bf16::from_f32(v)).collect() }}
pub fn reference_vector() -> Vec<bf16> {{ VEC_F32.iter().map(|&v| bf16::from_f32(v)).collect() }}
pub fn reference_expected() -> Vec<bf16> {{ EXP_F32.iter().map(|&v| bf16::from_f32(v)).collect() }}
"""
    return ref_rs


# 코드 생성 — dot_product (batch_gemm과 별도 구조 — 검증된 dot_product_kernel.rs 그대로 반영)
# =====================================================================

def _gen_kernel_dot_product(axis_size: int):
    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.dot_product — 검증된 dot_product_kernel.rs 구조 그대로 사용.
// batch_gemm 템플릿의 degenerate(M=1,N=1) 축소판이 하드웨어 제약(Time::SIZE가
// Lane::SIZE(8)로 나눠지지 않음)에 걸려 실패했기 때문에, 별도의 벡터 전체-reduction
// 전용 구조를 사용한다.
axes![A = {axis_size}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![1 # 256];
pub type Time = m![1];
pub type Lane = m![1];

#[device(chip = 1)]
pub fn pilot_e2e_dot_product_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<bf16, Chip, m![A]>,
    rhs: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![1]> {{
    let lhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = lhs.to_dm(&mut ctx.tdma, 0);
    let rhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = rhs.to_dm(&mut ctx.tdma, 1 << 12);

    let rhs: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![A]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![{{ Time }}, A / 16], m![A % 16]>()
        .to_trf(TrfAddress::Full);

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![A / 16], m![A % 16]>()
        .contract_outer::<m![A / 32], m![A % 32], _, _>(&rhs)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 8]>()
        .commit(1 << 13);

    result.to_hbm(&mut ctx.tdma, 2 << 28)
}}
"""

    host_and_test_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_dot_product_kernel::{{A, pilot_e2e_dot_product_kernel}};

mod reference_data_e2e_dot_product;
use reference_data_e2e_dot_product::{{reference_a, reference_b, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let lhs = HostTensor::<bf16, m![A]>::from_buf(reference_a());
    let rhs = HostTensor::<bf16, m![A]>::from_buf(reference_b());
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_e2e_dot_product_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Pilot E2E dot_product: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn matches_bf16_reference() {{
        let mut ctx = Context::acquire();
        let lhs = HostTensor::<bf16, m![A]>::from_buf(reference_a());
        let rhs = HostTensor::<bf16, m![A]>::from_buf(reference_b());
        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

        let out_hbm = launch(pilot_e2e_dot_product_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
        let actual_buf: Vec<bf16> = out_hbm.to_host::<m![1]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        if let Some(&actual) = actual_buf.first() {{
            println!("=== 값 비교 ===");
            println!("  expected={{:?}} actual={{:?}}", expected, actual);
            let diff = (f32::from(actual) - f32::from(expected)).abs();
            let tol = (0.05 * f32::from(expected).abs()).max(1.0);
            assert!(diff <= tol, "dot_product mismatch: expected {{expected:?}}, actual {{actual:?}}, diff {{diff}} > tol {{tol}}");
        }}
    }}
}}
"""
    return kernel_rs, host_and_test_rs


def _gen_reference_dot_product(a_bf16, b_bf16, expected_bf16, axis_size):
    a_full = a_bf16 + [0.0] * (axis_size - len(a_bf16))
    b_full = b_bf16 + [0.0] * (axis_size - len(b_bf16))

    def build_const(arr, const_name):
        values_str = ", ".join(f"{float(v):.8}_f32" for v in arr)
        return f"const {const_name}: [f32; {len(arr)}] = [{values_str}];\n"

    a_const = build_const(a_full, "A_F32_VALUES")
    b_const = build_const(b_full, "B_F32_VALUES")

    return (
        "// AUTO-GENERATED — expected는 실제 PyTorch 입력(bf16 양자화)의 dot product\n"
        "// 나머지 원소는 0으로 패딩 (reduction sum이라 0-패딩은 결과에 영향 없음)\n"
        "use furiosa_opt_std::prelude::*;\n\n"
        f"{a_const}"
        f"{b_const}"
        "pub fn reference_a() -> Vec<bf16> { A_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
        "pub fn reference_b() -> Vec<bf16> { B_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
        f"pub fn reference_expected() -> bf16 {{ bf16::from_f32({float(expected_bf16):.8}_f32) }}\n"
    )


# =====================================================================
# 실행
# =====================================================================

if __name__ == "__main__":
    import sys

    AXIS_SIZE = 2048
    OUT_DIR = Path("generated")
    OUT_DIR.mkdir(exist_ok=True)

    # 인자 없이 실행하면 5개 전부, 인자를 주면 그 연산만 생성한다.
    # 예: python e2e_pipeline.py add batch_gemm
    requested_ops = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    class ElementwiseBinary(torch.nn.Module):
        def __init__(self, op_name):
            super().__init__()
            self.op_name = op_name

        def forward(self, x, y):
            if self.op_name == "add":
                return x + y
            if self.op_name == "sub":
                return x - y
            if self.op_name == "mul":
                return x * y
            if self.op_name == "div":
                return x / y

    class UnaryElementwise(torch.nn.Module):
        def __init__(self, op_name):
            super().__init__()
            self.op_name = op_name

        def forward(self, x):
            if self.op_name == "rsqrt":
                return torch.rsqrt(x)
            if self.op_name == "sqrt":
                return torch.sqrt(x)
            if self.op_name == "exp":
                return torch.exp(x)
            if self.op_name == "sigmoid":
                return torch.sigmoid(x)
            if self.op_name == "silu":
                return torch.nn.functional.silu(x)
            if self.op_name == "silu":
                return torch.nn.functional.silu(x)
            if self.op_name == "log":
                return torch.log(x.abs() + 0.1)  # 정의역 보장
            if self.op_name == "tanh":
                return torch.tanh(x)
            if self.op_name == "sin":
                return torch.sin(x)
            if self.op_name == "cos":
                return torch.cos(x)
            if self.op_name == "erf":
                return torch.erf(x)
            if self.op_name == "pow2":
                return torch.pow(x, 2.0)

    class BatchMatmul(torch.nn.Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    class Matmul2D(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    class DotProduct(torch.nn.Module):
        # torch.dot(aten.dot)은 이 torch_mlir 버전에서 미지원 확인됨.
        # torch.matmul(1D, 1D)은 내적과 동일 시맨틱이며 linalg.dot으로 lowering됨 (실측 확인).
        def forward(self, x, y):
            return torch.matmul(x, y)

    class Gemv(torch.nn.Module):
        def forward(self, A, x):
            return torch.mv(A, x)

    # --- Family A ---
    elementwise_ops = [op for op in ("add", "sub", "mul", "div") if requested_ops is None or op in requested_ops]
    for op_name in elementwise_ops:
        print(f"\n{'='*70}\n [Family A] op_name = {op_name}\n{'='*70}")

        torch.manual_seed(123)
        model = ElementwiseBinary(op_name)
        x = torch.randn(4, 4)
        if op_name == "div":
            # 분모가 0 근처면 값이 폭발해 오차 비교(1e-4)가 무의미해짐 -> 절댓값 + 오프셋으로 0 회피
            y = torch.randn(4, 4).abs() + 0.5
        else:
            y = torch.randn(4, 4)
        with torch.no_grad():
            actual_output = model(x, y)

        compiled = torch_mlir.compile(model, (x, y), output_type="linalg-on-tensors")
        ir_text_before = str(compiled)
        ctx = Context()
        module = Module.parse(ir_text_before, ctx)
        ops_before = _list_op_names(module.operation)
        specs, ann_before, ann_after = rewrite_to_rngd(module)
        spec = specs[0] if specs else None
        ir_text_after = str(module)
        ops_after = _list_op_names(module.operation)
        print(f"spec: {spec}")
        assert spec is not None and spec.family == "elementwise" and spec.rngd_op == op_name

        kernel_rs, host_test_rs = _gen_kernel_elementwise(op_name, AXIS_SIZE)
        reference_rs = _gen_reference_elementwise(
            x.flatten().tolist(), y.flatten().tolist(), actual_output.flatten().tolist(), AXIS_SIZE
        )

        prefix = f"e2e_{op_name}"
        with open(OUT_DIR / f"{prefix}_kernel.rs", "w") as f:
            f.write(kernel_rs)
        with open(OUT_DIR / f"{prefix}_host_test.rs", "w") as f:
            f.write(host_test_rs)
        with open(OUT_DIR / f"{prefix}_reference_data.rs", "w") as f:
            f.write(reference_rs)
        with open(OUT_DIR / f"{prefix}_ir_before.mlir", "w") as f:
            f.write(ir_text_before)
        with open(OUT_DIR / f"{prefix}_ir_after.mlir", "w") as f:
            f.write(ir_text_after)
        with open(OUT_DIR / f"{prefix}_ir_before_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_before, ann_before))
        with open(OUT_DIR / f"{prefix}_ir_after_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_after, ann_after))
        with open(OUT_DIR / f"{prefix}_ir_diff.txt", "w") as f:
            f.write(_diff_summary(ops_before, ops_after))
        print(f"생성 완료: {prefix}_*.rs, {prefix}_ir_before.mlir, {prefix}_ir_after.mlir, {prefix}_ir_diff.txt")

    # --- Family A (단항/unary) ---
    unary_elementwise_ops = [op for op in ("rsqrt", "sqrt", "exp", "sigmoid", "silu", "log", "tanh", "sin", "cos", "erf", "pow2") if requested_ops is None or op in requested_ops]
    for op_name in unary_elementwise_ops:
        print(f"\n{'='*70}\n [Family A-unary] op_name = {op_name}\n{'='*70}")

        torch.manual_seed(123)
        model = UnaryElementwise(op_name)
        if op_name in ("rsqrt", "sqrt"):
            x = torch.rand(4, 4) + 0.5  # rsqrt/sqrt 정의역(양수) 보장
        else:
            x = torch.randn(4, 4)  # pow2는 정의역 제한 없음
        with torch.no_grad():
            actual_output = model(x)

        compiled = torch_mlir.compile(model, (x,), output_type="linalg-on-tensors")
        ir_text_before = str(compiled)
        ctx = Context()
        module = Module.parse(ir_text_before, ctx)
        ops_before = _list_op_names(module.operation)
        specs, ann_before, ann_after = rewrite_to_rngd(module)
        spec = specs[0] if specs else None
        ir_text_after = str(module)
        ops_after = _list_op_names(module.operation)
        print(f"spec: {spec}")
        assert spec is not None and spec.family == "elementwise" and spec.rngd_op == op_name

        if op_name == "pow2":
            kernel_rs, host_test_rs = _gen_kernel_elementwise_square(AXIS_SIZE)
        else:
            kernel_rs, host_test_rs = _gen_kernel_elementwise_unary(op_name, AXIS_SIZE)
        reference_rs = _gen_reference_elementwise_unary(
            x.flatten().tolist(), actual_output.flatten().tolist(), AXIS_SIZE
        )

        prefix = f"e2e_{op_name}"
        with open(OUT_DIR / f"{prefix}_kernel.rs", "w") as f:
            f.write(kernel_rs)
        with open(OUT_DIR / f"{prefix}_host_test.rs", "w") as f:
            f.write(host_test_rs)
        with open(OUT_DIR / f"{prefix}_reference_data.rs", "w") as f:
            f.write(reference_rs)
        with open(OUT_DIR / f"{prefix}_ir_before.mlir", "w") as f:
            f.write(ir_text_before)
        with open(OUT_DIR / f"{prefix}_ir_after.mlir", "w") as f:
            f.write(ir_text_after)
        with open(OUT_DIR / f"{prefix}_ir_before_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_before, ann_before))
        with open(OUT_DIR / f"{prefix}_ir_after_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_after, ann_after))
        with open(OUT_DIR / f"{prefix}_ir_diff.txt", "w") as f:
            f.write(_diff_summary(ops_before, ops_after))
        print(f"생성 완료: {prefix}_*.rs, {prefix}_ir_before.mlir, {prefix}_ir_after.mlir, {prefix}_ir_diff.txt")



    # --- Family B ---
    contraction_cases_all = [
        # (op_name, model, seed, x_shape, y_shape)
        ("batch_gemm", BatchMatmul(), 42, (32, 32, 32), (32, 32, 8)),
        ("gemm", Matmul2D(), 43, (32, 32), (32, 8)),
        ("dot_product", DotProduct(), 44, (32,), (32,)),
        ("gemv", Gemv(), 45, (256, 2048), (2048,)),
    ]
    contraction_cases = [
        c for c in contraction_cases_all if requested_ops is None or c[0] in requested_ops
    ]

    for op_name, model, seed, x_shape, y_shape in contraction_cases:
        print(f"\n{'='*70}\n [Family B] op_name = {op_name}\n{'='*70}")

        torch.manual_seed(seed)
        x = torch.randn(*x_shape)
        y = torch.randn(*y_shape)

        compiled = torch_mlir.compile(model, (x, y), output_type="linalg-on-tensors")
        ir_text_before = str(compiled)
        ctx = Context()
        module = Module.parse(ir_text_before, ctx)
        ops_before = _list_op_names(module.operation)
        specs, ann_before, ann_after = rewrite_to_rngd(module)
        spec = specs[0] if specs else None
        ir_text_after = str(module)
        ops_after = _list_op_names(module.operation)
        print(f"spec: {spec}")
        assert spec is not None and spec.family == "contraction" and spec.rngd_op == op_name

        # 배치 축이 없으면(2D gemm) V=1로 취급 — batch_gemm과 동일 검증 구조 재사용
        v = spec.batch if spec.batch is not None else 1

        x_bf16_f32 = x.to(torch.bfloat16).to(torch.float32)
        y_bf16_f32 = y.to(torch.bfloat16).to(torch.float32)
        with torch.no_grad():
            expected_f32_acc = model(x_bf16_f32, y_bf16_f32)
        expected_bf16 = expected_f32_acc.to(torch.bfloat16)

        kernel_rs, host_test_rs = _gen_kernel_batch_gemm(v, spec.m, spec.k, spec.n, op_name=op_name)
        reference_rs = _gen_reference_batch_gemm(
            x_bf16_f32.flatten().tolist(),
            y_bf16_f32.flatten().tolist(),
            expected_bf16.to(torch.float32).flatten().tolist(),
        )

        if op_name == "dot_product":
            # batch_gemm 템플릿의 degenerate(M=1,N=1) 케이스가 하드웨어 제약에 걸려
            # 실패했으므로, 검증된 dot_product_kernel.rs 구조를 그대로 쓰는 전용 함수로 대체.
            kernel_rs, host_test_rs = _gen_kernel_dot_product(AXIS_SIZE)
            reference_rs = _gen_reference_dot_product(
                x_bf16_f32.flatten().tolist(),
                y_bf16_f32.flatten().tolist(),
                expected_bf16.to(torch.float32).item(),
                AXIS_SIZE,
            )

        elif op_name == "gemv":
            # gemv_kernel.rs 구조 재사용
            I_SIZE = x_shape[0]  # 256
            J_SIZE = x_shape[1]  # 2048
            kernel_rs = _gen_kernel_gemv(I_SIZE, J_SIZE)
            host_test_rs = _gen_pilot_gemv(I_SIZE, J_SIZE)
            mat_bf16 = x.to(torch.bfloat16)
            vec_bf16 = y.to(torch.bfloat16)
            expected_gemv = torch.mv(
                mat_bf16.to(torch.float32),
                vec_bf16.to(torch.float32)
            ).to(torch.bfloat16)
            reference_rs = _gen_reference_gemv(mat_bf16, vec_bf16, expected_gemv)

        prefix = f"e2e_{op_name}"
        with open(OUT_DIR / f"{prefix}_kernel.rs", "w") as f:
            f.write(kernel_rs)
        with open(OUT_DIR / f"{prefix}_host_test.rs", "w") as f:
            f.write(host_test_rs)
        with open(OUT_DIR / f"{prefix}_reference_data.rs", "w") as f:
            f.write(reference_rs)
        with open(OUT_DIR / f"{prefix}_ir_before.mlir", "w") as f:
            f.write(ir_text_before)
        with open(OUT_DIR / f"{prefix}_ir_after.mlir", "w") as f:
            f.write(ir_text_after)
        with open(OUT_DIR / f"{prefix}_ir_before_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_before, ann_before))
        with open(OUT_DIR / f"{prefix}_ir_after_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_after, ann_after))
        with open(OUT_DIR / f"{prefix}_ir_diff.txt", "w") as f:
            f.write(_diff_summary(ops_before, ops_after))
        print(f"생성 완료: {prefix}_*.rs, {prefix}_ir_before.mlir, {prefix}_ir_after.mlir, {prefix}_ir_diff.txt")




# ══════════════════════════════════════════════════════════════════
# 메타데이터 마커 — 변환 후 잔류 op 분류
# rewrite_to_rngd() 이후 별도 패스로 실행.
# 각 잔류 op에 "왜 남았고 어떻게 처리해야 하는지" 정보를 부여한다.
# ══════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════
# 데이터 흐름 추출 — rewrite_to_rngd() 이후 rngd.* op 간 SSA 그래프
# 코드 생성기(Phase B)의 입력으로 사용됨.
# ══════════════════════════════════════════════════════════════════

@dataclass
class DfNode:
    """데이터 흐름 그래프의 노드 하나 — rngd.* op 1개에 대응"""
    idx: int                    # 토폴로지 순서 인덱스
    ssa_name: str               # SSA 값 이름 (%8, %9, ...)
    op_type: str                # rngd.batch_gemm, rngd.elementwise, rngd.reduce
    op_attr: str                # 세부 연산 (sigmoid, mul, sum, ...)
    attrs: dict                 # 추가 속성 (transpose_rhs 등)
    in_shape: list              # 입력 shape들
    out_shape: list             # 출력 shape
    inputs: list                # 입력 SSA 이름 리스트
    input_sources: list         # 각 입력의 출처: "rngd" | "arg" | "const" | "other"
    dtype: str                  # 출력 dtype (f32, bf16 등)


def extract_dataflow(module: Module) -> list:
    """rewrite_to_rngd() 이후 모듈에서 rngd.* op의 데이터 흐름 그래프를 추출.
    반환: DfNode 리스트 (토폴로지 순서, 즉 실행 순서)"""
    nodes = []
    rngd_results = set()  # rngd.* op이 생성한 SSA 값 이름들

    def _ssa_name(value) -> str:
        """MLIR Value의 SSA 이름 추출 (%0, %1, ...)"""
        s = str(value)
        m = re.search(r"%(\d+)", s)
        if m:
            return "%" + m.group(1)
        if "block argument" in s.lower() or "arg" in s[:30].lower():
            return "%arg"
        return s[:15]

    def _result_ssa(op) -> str:
        """op의 첫 번째 result SSA 이름"""
        if len(op.results) > 0:
            return _ssa_name(op.results[0])
        return "??"

    def _input_source(value, rngd_set) -> str:
        """입력 값의 출처 분류"""
        name = _ssa_name(value)
        if name in rngd_set:
            return "rngd"
        owner = value.owner
        if owner is None or not hasattr(owner, "name"):
            return "arg"  # block argument = 함수 인자
        owner_name = str(owner.name)
        if "arith.constant" in owner_name or "ml_program" in owner_name:
            return "const"
        return "other"

    def _shape_list(value) -> list:
        """Value의 텐서 shape을 리스트로"""
        try:
            return list(RankedTensorType(value.type).shape)
        except:
            return []

    def _dtype_str(value) -> str:
        """Value의 element type 문자열"""
        try:
            t = RankedTensorType(value.type)
            return str(t.element_type)
        except:
            return "??"

    def _get_attr(op, key) -> str:
        """op의 속성값을 문자열로"""
        try:
            return str(op.attributes[key]).strip('"')
        except:
            return ""

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    name = str(inner.name)
                    if name.startswith("rngd."):
                        ssa = _result_ssa(inner)
                        rngd_results.add(ssa)

                        # 속성 추출
                        op_attr = _get_attr(inner, "op")
                        extra_attrs = {}
                        for attr_name in ("transpose_rhs",):
                            v = _get_attr(inner, attr_name)
                            if v:
                                extra_attrs[attr_name] = v

                        # 입력 분석
                        input_names = []
                        input_sources = []
                        in_shapes = []
                        for operand in inner.operands:
                            input_names.append(_ssa_name(operand))
                            input_sources.append(_input_source(operand, rngd_results))
                            in_shapes.append(_shape_list(operand))

                        node = DfNode(
                            idx=len(nodes),
                            ssa_name=ssa,
                            op_type=name,
                            op_attr=op_attr,
                            attrs=extra_attrs,
                            in_shape=in_shapes,
                            out_shape=_shape_list(inner.results[0]) if inner.results else [],
                            inputs=input_names,
                            input_sources=input_sources,
                            dtype=_dtype_str(inner.results[0]) if inner.results else "??",
                        )
                        nodes.append(node)
                    else:
                        walk(inner)

    walk(module.operation)
    return nodes


def print_dataflow(nodes: list):
    """데이터 흐름 그래프를 사람이 읽을 수 있는 형태로 출력"""
    print(f"=== 데이터 흐름 그래프 ({len(nodes)}개 rngd.* op) ===")
    for n in nodes:
        attr_str = f"({n.op_attr})" if n.op_attr else ""
        extra = " ".join(f"{k}={v}" for k, v in n.attrs.items())
        if extra:
            attr_str += f" [{extra}]"

        print(f"\n  [{n.idx}] {n.ssa_name} = {n.op_type}{attr_str}")
        print(f"      출력: {n.out_shape} ({n.dtype})")
        for i, (inp, src, shp) in enumerate(zip(n.inputs, n.input_sources, n.in_shape)):
            marker = "←rngd" if src == "rngd" else f"←{src}"
            print(f"      입력[{i}]: {inp} {shp} {marker}")




# ══════════════════════════════════════════════════════════════════
# Fusion 패턴 마킹 — extract_dataflow() 결과에서 복합 패턴 감지
# 개별 rngd.* op은 유지하면서 "이 그룹이 무엇인지" 메타데이터만 부여.
# 백엔드 컴파일러가 fused kernel 최적화 여부를 선택할 수 있도록 힌트 제공.
# ══════════════════════════════════════════════════════════════════

@dataclass
class FusionGroup:
    """감지된 fusion 패턴 하나"""
    name: str           # RMSNORM, SOFTMAX, SILU, FFN, RESIDUAL 등
    node_indices: list   # DfNode.idx 리스트
    description: str     # 사람이 읽을 수 있는 설명


def detect_fusion_patterns(nodes: list) -> list:
    """DfNode 리스트에서 알려진 fusion 패턴을 감지.
    반환: FusionGroup 리스트. 하나의 노드가 여러 그룹에 속할 수 있음 (예: SiLU는 FFN의 일부)."""
    groups = []
    assigned = {}  # idx → 소속 그룹 이름 (중첩 허용을 위해 리스트로도 가능)

    # SSA → idx 매핑
    ssa_to_idx = {n.ssa_name: n.idx for n in nodes}
    idx_to_node = {n.idx: n for n in nodes}

    def _follows(node, expected_source_idx):
        """node의 rngd 입력 중 하나가 expected_source_idx인지"""
        for inp, src in zip(node.inputs, node.input_sources):
            if src == "rngd" and ssa_to_idx.get(inp) == expected_source_idx:
                return True
        return False

    def _attr(idx):
        return idx_to_node[idx].op_attr if idx in idx_to_node else ""

    # ── RMSNorm 감지 ──
    # 패턴: pow2 → sum → div → add → rsqrt → mul → mul
    for n in nodes:
        if _attr(n.idx) != "pow2":
            continue
        try:
            chain = [n.idx]  # [pow2]
            # pow2 → sum
            sum_candidates = [m for m in nodes if _attr(m.idx) in ("sum","reduce_sum") and _follows(m, chain[-1])]
            if not sum_candidates:
                continue
            chain.append(sum_candidates[0].idx)
            # sum → div
            div_candidates = [m for m in nodes if _attr(m.idx) == "div" and _follows(m, chain[-1])]
            if not div_candidates:
                continue
            chain.append(div_candidates[0].idx)
            # div → add (eps)
            add_candidates = [m for m in nodes if _attr(m.idx) == "add" and _follows(m, chain[-1])]
            if not add_candidates:
                continue
            chain.append(add_candidates[0].idx)
            # add → rsqrt
            rsqrt_candidates = [m for m in nodes if _attr(m.idx) == "rsqrt" and _follows(m, chain[-1])]
            if not rsqrt_candidates:
                continue
            chain.append(rsqrt_candidates[0].idx)
            # rsqrt → mul (x * rsqrt)
            mul1_candidates = [m for m in nodes if _attr(m.idx) == "mul" and _follows(m, chain[-1])]
            if not mul1_candidates:
                continue
            chain.append(mul1_candidates[0].idx)
            # mul → mul (result * weight)
            mul2_candidates = [m for m in nodes if _attr(m.idx) == "mul" and _follows(m, chain[-1])]
            if not mul2_candidates:
                continue
            chain.append(mul2_candidates[0].idx)

            group_num = sum(1 for g in groups if g.name == "RMSNORM") + 1
            groups.append(FusionGroup(
                name="RMSNORM",
                node_indices=chain,
                description=f"RMSNorm #{group_num}: pow2→sum→div→add(eps)→rsqrt→mul(x)→mul(weight) — 7 ops → 1 fused op 가능",
            ))
        except (IndexError, KeyError):
            continue

    # ── Softmax 감지 ──
    # 패턴: max → sub → exp → sum → div
    for n in nodes:
        if _attr(n.idx) not in ("max", "reduce_max"):
            continue
        try:
            chain = [n.idx]  # [max]
            # max → sub
            sub_candidates = [m for m in nodes if _attr(m.idx) == "sub" and _follows(m, chain[-1])]
            if not sub_candidates:
                continue
            chain.append(sub_candidates[0].idx)
            # sub → exp
            exp_candidates = [m for m in nodes if _attr(m.idx) == "exp" and _follows(m, chain[-1])]
            if not exp_candidates:
                continue
            chain.append(exp_candidates[0].idx)
            # exp → sum
            sum_candidates = [m for m in nodes if _attr(m.idx) in ("sum","reduce_sum") and _follows(m, chain[-1])]
            if not sum_candidates:
                continue
            chain.append(sum_candidates[0].idx)
            # sum → div
            div_candidates = [m for m in nodes if _attr(m.idx) == "div" and _follows(m, chain[-1])]
            if not div_candidates:
                continue
            chain.append(div_candidates[0].idx)

            groups.append(FusionGroup(
                name="SOFTMAX",
                node_indices=chain,
                description="Softmax: max→sub(x-max)→exp→sum→div — 5 ops → 1 fused op 가능",
            ))
        except (IndexError, KeyError):
            continue

    # ── SiLU 감지 ──
    # 패턴: sigmoid → mul (여기서 mul의 두 입력이 sigmoid 출력 + sigmoid 입력)
    for n in nodes:
        if _attr(n.idx) != "sigmoid":
            continue
        sig_idx = n.idx
        sig_ssa = n.ssa_name
        # sigmoid 입력 — rngd 또는 other (FX 경로에서 layout op 경유 시 other)
        sig_input_ssa = [inp for inp in n.inputs]
        if not sig_input_ssa:
            continue
        sig_input = sig_input_ssa[0]
        # sigmoid 출력을 사용하는 mul 찾기
        for m in nodes:
            if _attr(m.idx) == "mul" and sig_ssa in m.inputs and sig_input in m.inputs:
                groups.append(FusionGroup(
                    name="SILU",
                    node_indices=[sig_idx, m.idx],
                    description=f"SiLU: sigmoid(x)·x — 2 ops → 1 fused activation",
                ))
                break

    # ── Residual Add 감지 ──
    # 패턴 1: add에서 한쪽 입력이 arg (함수 인자) → 첫 번째 residual
    # 패턴 2: add에서 두 입력 모두 rngd이지만, 한쪽이 훨씬 이전 노드 → skip connection
    for n in nodes:
        if _attr(n.idx) != "add":
            continue
        # 패턴 1: arg 입력
        if any(src == "arg" for src in n.input_sources):
            groups.append(FusionGroup(
                name="RESIDUAL",
                node_indices=[n.idx],
                description="Residual connection: x + sublayer(x)",
            ))
            continue
        # 패턴 2: 입력 중 하나가 rngd이고, 두 입력 간 노드 간격이 5 이상 → skip connection
        all_inputs = [(inp, ssa_to_idx.get(inp, -1))
                      for inp in n.inputs]
        has_rngd = any(src == "rngd" for src in n.input_sources)
        if has_rngd and len(all_inputs) == 2:
            idx_a, idx_b = all_inputs[0][1], all_inputs[1][1]
            if idx_a >= 0 and idx_b >= 0 and abs(idx_a - idx_b) >= 5:
                groups.append(FusionGroup(
                    name="RESIDUAL",
                    node_indices=[n.idx],
                    description=f"Residual connection: skip {abs(idx_a - idx_b)} nodes",
                ))
            elif has_rngd and n.idx > 10:
                # rngd 입력이 있고 블록 후반부의 add → residual로 추정
                rngd_inp_idx = [ssa_to_idx.get(inp, -1) for inp, src in zip(n.inputs, n.input_sources) if src == "rngd"]
                if rngd_inp_idx and (n.idx - max(rngd_inp_idx)) >= 5:
                    groups.append(FusionGroup(
                        name="RESIDUAL",
                        node_indices=[n.idx],
                        description=f"Residual connection: rngd+other skip",
                    ))

    # ── Attention 전체 감지 ──
    # 패턴: 연속된 batch_gemm 3개(QKV) + score 연산 + Softmax + batch_gemm(AV) + batch_gemm(O_proj)
    # QKV는 같은 외부 입력을 공유하는 batch_gemm 3개로 식별
    gemm_nodes = [n for n in nodes if n.op_type in ("rngd.batch_gemm", "rngd.gemm")]
    for i, gn in enumerate(gemm_nodes):
        # 같은 입력(other/arg)을 공유하는 연속 gemm 3개 찾기
        if i + 2 >= len(gemm_nodes):
            continue
        g1, g2, g3 = gemm_nodes[i], gemm_nodes[i+1], gemm_nodes[i+2]
        # 3개가 연속 인덱스이고 같은 입력을 하나 이상 공유
        shared_inputs = set()
        g1_inputs = set(inp for inp, src in zip(g1.inputs, g1.input_sources))
        g2_inputs = set(inp for inp, src in zip(g2.inputs, g2.input_sources))
        shared_inputs = g1_inputs & g2_inputs
        if not shared_inputs:
            continue
        # g3도 같은 입력 공유 확인
        g3_inputs = set(inp for inp, src in zip(g3.inputs, g3.input_sources))
        g3_shares = bool(shared_inputs & g3_inputs)
        if not g3_shares:
            continue

        # QKV 이후 Softmax까지의 범위 찾기
        qkv_start = g1.idx
        # 이미 감지된 SOFTMAX가 QKV 뒤에 있는지
        softmax_group = None
        for sg in groups:
            if sg.name == "SOFTMAX" and min(sg.node_indices) > qkv_start:
                softmax_group = sg
                break
        if not softmax_group:
            continue

        # Softmax 이후 ~ 첫 번째 RESIDUAL 직전이 Attention 범위
        softmax_end = max(softmax_group.node_indices)

        # 첫 번째 RESIDUAL add를 찾아 그 직전에서 끊기
        attn_end = None
        for rg in groups:
            if rg.name == "RESIDUAL" and min(rg.node_indices) > softmax_end:
                attn_end = min(rg.node_indices) - 1
                break
        if attn_end is None:
            # RESIDUAL 없으면 softmax 이후 첫 gemm(o_proj)까지
            post_gemms = [gn for gn in gemm_nodes if gn.idx > softmax_end]
            if not post_gemms:
                continue
            attn_end = post_gemms[0].idx

        attn_indices = [n.idx for n in nodes if qkv_start <= n.idx <= attn_end]

        groups.append(FusionGroup(
            name="ATTENTION",
            node_indices=attn_indices,
            description=f"Multi-Head Attention: QKV proj → score(QK^T·scale) → Softmax → AV → O proj — {len(attn_indices)} ops",
        ))
        break  # Block당 1개

    # ── FFN 전체 감지 ──
    # 패턴: batch_gemm(gate) → sigmoid → mul(silu) → batch_gemm(up) → mul(gate*up) → batch_gemm(down)
    # SiLU 그룹이 이미 감지됐으면, 그 주변의 gemm 3개를 포함
    for sg in groups:
        if sg.name != "SILU":
            continue
        silu_min = min(sg.node_indices)
        silu_max = max(sg.node_indices)

        # SiLU 앞의 gemm (gate)
        gate_gemm = None
        for n in reversed(nodes):
            if n.idx < silu_min and n.op_type in ("rngd.batch_gemm", "rngd.gemm"):
                gate_gemm = n
                break
        if not gate_gemm:
            continue

        # SiLU 뒤의 연산들: up_gemm, gate*up mul, down_gemm
        post_silu = [n for n in nodes if n.idx > silu_max]
        up_gemm = None
        gate_up_mul = None
        down_gemm = None
        for n in post_silu:
            if n.op_type in ("rngd.batch_gemm", "rngd.gemm") and up_gemm is None:
                up_gemm = n
            elif n.op_attr == "mul" and gate_up_mul is None and up_gemm is not None:
                gate_up_mul = n
            elif n.op_type in ("rngd.batch_gemm", "rngd.gemm") and gate_up_mul is not None:
                down_gemm = n
                break

        if not all([gate_gemm, up_gemm, gate_up_mul, down_gemm]):
            continue

        ffn_indices = [n.idx for n in nodes if gate_gemm.idx <= n.idx <= down_gemm.idx]
        groups.append(FusionGroup(
            name="FFN",
            node_indices=ffn_indices,
            description=f"FFN (SwiGLU): gate→SiLU→up→gate·up→down — {len(ffn_indices)} ops",
        ))

    return groups


def print_fusion_report(nodes: list, groups: list):
    """fusion 감지 결과 출력"""
    print(f"=== Fusion 패턴 감지 ({len(groups)}개 그룹) ===")
    for g in groups:
        ssa_list = ", ".join(idx_to_node[i].ssa_name for i in g.node_indices
                           if (idx_to_node := {n.idx: n for n in nodes}) and i in idx_to_node)
        ops = " → ".join(idx_to_node[i].op_attr or idx_to_node[i].op_type.replace("rngd.","")
                         for i in g.node_indices
                         if (idx_to_node := {n.idx: n for n in nodes}) and i in idx_to_node)
        print(f"\n  [{g.name}] {g.description}")
        print(f"    노드: {g.node_indices}")
        print(f"    체인: {ops}")

    # 미소속 노드 표시
    all_fused = set()
    for g in groups:
        all_fused.update(g.node_indices)
    unfused = [n for n in nodes if n.idx not in all_fused]
    if unfused:
        print(f"\n  [미소속] {len(unfused)}개 노드")
        for n in unfused:
            attr = n.op_attr or n.op_type.replace("rngd.","")
            print(f"    [{n.idx}] {n.ssa_name} {attr}")


MARKER_KINDS = {
    "CONVERTED":         "rngd.* 변환 완료",
    "LAYOUT_TRANSFORM":  "데이터 형태 변환 (reshape/expand/broadcast) — 실행 불필요, 메타데이터만 변경",
    "ABSORBABLE":        "후속 패스에서 인접 op에 흡수 가능 (fill, constant 등)",
    "TYPE_CAST":         "타입 변환 (f32<->bf16) — rngd.cast로 변환 예정",
    "INFRASTRUCTURE":    "모듈 구조 (함수 정의, 가중치 저장소) — 변환 대상 아님",
    "CPU_FALLBACK":      "NPU 미지원 — CPU 처리 필요",
    "UNKNOWN":           "미분류 — 수동 확인 필요",
}


def mark_remaining(module: Module) -> dict:
    """rewrite_to_rngd() 이후 잔류 op들을 분류하여 마커를 부여한다.
    반환: {marker_kind: [(op_name, reason)]} 딕셔너리"""
    from collections import defaultdict
    report = defaultdict(list)

    def classify(op):
        name = str(op.name)

        if name.startswith("rngd."):
            report["CONVERTED"].append((name, ""))
            return
        if name in ("func.func", "func.return", "ml_program.global", "builtin.module") or name.startswith('"'):
            report["INFRASTRUCTURE"].append((name, "모듈 구조"))
            return
        if name.startswith("ml_program."):
            report["INFRASTRUCTURE"].append((name, "가중치 저장소"))
            return
        if name in ("tensor.empty", "tensor.expand_shape", "tensor.collapse_shape",
                     "tensor.cast", "tensor.extract_slice", "tensor.insert_slice",
                     "tensor.concat"):
            report["LAYOUT_TRANSFORM"].append((name, "텐서 구조 변환"))
            return
        if name == "linalg.fill":
            report["ABSORBABLE"].append((name, "contraction/reduce 초기화"))
            return
        if name == "arith.constant":
            report["ABSORBABLE"].append((name, "상수값"))
            return
        if name == "linalg.generic":
            try:
                body = op.regions[0].blocks[0]
                body_ops = [str(inner.name) for inner in body.operations]
                non_yield = [b for b in body_ops if b != "linalg.yield"]
                if len(non_yield) == 0:
                    report["LAYOUT_TRANSFORM"].append((name, "identity generic — reshape/broadcast"))
                    return
                if any("truncf" in b or "extf" in b for b in non_yield):
                    report["TYPE_CAST"].append((name, f"타입 변환: {non_yield}"))
                    return
                if non_yield == ["arith.negf"]:
                    report["CONVERTED"].append((name, "negf generic → rngd.elementwise(negf) 변환 대상"))
                    return
                report["UNKNOWN"].append((name, f"미분류 body: {non_yield}"))
                return
            except (RuntimeError, ValueError):
                pass
        if name == "linalg.yield":
            report["ABSORBABLE"].append((name, "parent generic에 종속"))
            return
        if name == "arith.negf":
            report["CONVERTED"].append((name, "negf → rngd.elementwise(negf) 변환 대상"))
            return
        if name.startswith(("arith.", "math.")):
            report["CPU_FALLBACK"].append((name, "단독 잔류 arith/math op"))
            return
        report["UNKNOWN"].append((name, ""))

    def walk(op):
        classify(op)
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)

    walk(module.operation)
    return dict(report)


def print_marker_report(report: dict):
    """마커 분류 결과를 요약 출력."""
    total = sum(len(v) for v in report.values())
    print(f"=== 메타데이터 마커 분류 (총 {total}개 op) ===")
    for kind in ["CONVERTED", "LAYOUT_TRANSFORM", "ABSORBABLE", "TYPE_CAST",
                  "INFRASTRUCTURE", "CPU_FALLBACK", "UNKNOWN"]:
        items = report.get(kind, [])
        if not items:
            continue
        desc = MARKER_KINDS[kind]
        print(f"\n  [{kind}] {desc} -- {len(items)}개")
        from collections import Counter
        counts = Counter(name for name, _ in items)
        for op_name, cnt in counts.most_common():
            reason = next((r for n, r in items if n == op_name and r), "")
            suffix = f" ({reason})" if reason else ""
            print(f"    {op_name}: {cnt}개{suffix}")


# ──────────────────────────────────────────────────────────────────
# 타일링 패스 PoC (기존 코드 변경 없음 — 신규 추가)
# linalg.matmul(K=K_total, N=N_total) →
#   pilot_e2e_gemm_kernel(K=16, N=8) × (K_total/16) × (N_total/8)
# ──────────────────────────────────────────────────────────────────

def gen_tiled_linear_host(
        k_total: int,
        n_total: int,
        k_tile: int = 16,
        n_tile: int = 8,
        op_name: str = "tiled_gemm",
) -> str:
    """
    K_total, N_total 크기의 matmul을 k_tile × n_tile 커널로 타일링하는
    host 테스트 코드를 생성한다.

    사용하는 커널: pilot_e2e_gemm_kernel (K=16, N=8) — 기존 코드 재사용
    누적(accumulate): pilot_e2e_add_kernel의 add 패턴 (f32 elementwise)
    """
    assert k_total % k_tile == 0, f"K_total={k_total}이 k_tile={k_tile}의 배수여야 함"
    assert n_total % n_tile == 0, f"N_total={n_total}이 n_tile={n_tile}의 배수여야 함"

    k_tiles = k_total // k_tile
    n_tiles = n_total // n_tile
    total_kernels = k_tiles * n_tiles

    k_tiles = k_total // k_tile
    n_tiles = n_total // n_tile

    # 테스트용 reference data 인라인 생성
    import torch
    torch.manual_seed(42)
    x = torch.randn(1, k_total).to(torch.bfloat16)
    w = torch.randn(n_total, k_total).to(torch.bfloat16)
    out = torch.nn.functional.linear(x, w)

    def fmt(lst): return ", ".join(f"{v:.8f}_f32" for v in lst)
    def tolist(t): return t.flatten().to(torch.float32).tolist()

    x_data = fmt(tolist(x))
    expected_data = fmt(tolist(out))
    expected_norm = float(out.to(torch.float32).norm())

    # 타일별 가중치 데이터 (f32로 저장, 런타임에 bf16 변환)
    tile_consts = []
    for n in range(n_tiles):
        for k in range(k_tiles):
            w_tile = w[n*n_tile:(n+1)*n_tile, k*k_tile:(k+1)*k_tile]
            tile_consts.append(
                f"const W_N{n}_K{k}_F32: &[f32] = &[{fmt(tolist(w_tile))}];"
            )

    tile_consts_str = "\n".join(tile_consts)

    # 테스트 내 x_tiles, w_tiles 구성
    x_tile_lines = "\n".join([
        f"            x_tiles[{k}] = X_DATA[{k*k_tile}..{(k+1)*k_tile}].iter().map(|&v| bf16::from_f32(v)).collect();"
        for k in range(k_tiles)
    ])
    w_tile_lines = "\n".join([
        f"            w_tiles[{n}][{k}] = W_N{n}_K{k}_F32.iter().map(|&v| bf16::from_f32(v)).collect();"
        for n in range(n_tiles)
        for k in range(k_tiles)
    ])

    return f"""// AUTO-GENERATED tiled linear host — PoC
// linalg.matmul(M=1, K={k_total}, N={n_total})
// → pilot_golden_ffn_gate_kernel(V=1, M=1, K=16, N=8) × {k_tiles} × {n_tiles} = {total_kernels}회
//
// 타일링 전략:
//   N 방향: N={n_total}을 N={n_tile}씩 {n_tiles}개로 분할 → 결과 concat (오차 없음)
//   K 방향: K={k_total}을 K={k_tile}씩 {k_tiles}개로 분할 → 부분합 f32 accumulate
//
// ⚠️  성능 주의: Naive 타일링 — HBM 왕복 {total_kernels * 3}회
//    실제 최적화는 커널 내부 K 루프 또는 vector_inter_slice_reduce API 필요

use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_ffn_gate_kernel::{{
    V, M as GM, K as GK, N as GN, pilot_golden_ffn_gate_kernel
}};

/// 타일링된 matmul: x[1, {k_total}] × w[{n_total}, {k_total}] → out[1, {n_total}]
/// w_tiles: [n_tile_idx][k_tile_idx] 순서로 bf16 데이터
pub async fn tiled_matmul_{k_total}x{n_total}(
    ctx: &mut Context,
    x_tiles: &[Vec<bf16>],              // K 방향 분할 입력 [{k_tiles}개, 각 K={k_tile}]
    w_tiles: &[Vec<Vec<bf16>>],         // [N타일][K타일] 가중치
) -> Vec<f32> {{
    let mut out = Vec::with_capacity({n_total});

    for n in 0..{n_tiles} {{
        // K 방향 부분합 accumulate
        let mut acc: Vec<f32> = vec![0.0_f32; {n_tile}];

        for k in 0..{k_tiles} {{
            let x_hbm = HostTensor::<bf16, m![V, GM, GK]>::from_buf(x_tiles[k].clone())
                .to_hbm(&mut ctx.pdma, ((n * {k_tiles} * 2 + k * 2) as u64) << 18).await;
            let w_hbm = HostTensor::<bf16, m![V, GN, GK]>::from_buf(w_tiles[n][k].clone())
                .to_hbm(&mut ctx.pdma, ((n * {k_tiles} * 2 + k * 2 + 1) as u64) << 18).await;

            let p_hbm = launch(pilot_golden_ffn_gate_kernel, (ctx, &x_hbm, &w_hbm)).await;
            let p_f32: Vec<f32> = p_hbm
                .to_host::<m![V, GM, GN]>(&mut ctx.pdma).await.to_buf()
                .iter().map(|v: &bf16| v.to_f32()).collect();

            // f32 accumulate (CPU — 정밀도 유지)
            for i in 0..{n_tile} {{
                acc[i] += p_f32[i];
            }}
        }}
        out.extend(acc);
    }}
    out
}}

const X_DATA: &[f32] = &[{x_data}];
const EXPECT: &[f32] = &[{expected_data}];
{tile_consts_str}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn tiled_matmul_k{k_total}_n{n_total}_test() {{
        let mut ctx = Context::acquire();

        let mut x_tiles: Vec<Vec<bf16>> = vec![vec![]; {k_tiles}];
        let mut w_tiles: Vec<Vec<Vec<bf16>>> = vec![vec![vec![]; {k_tiles}]; {n_tiles}];

        {x_tile_lines}
        {w_tile_lines}

        let out = tiled_matmul_{k_total}x{n_total}(&mut ctx, &x_tiles, &w_tiles).await;

        let exp_norm: f32 = EXPECT.iter().map(|v| v*v).sum::<f32>().sqrt();
        let out_norm: f32 = out.iter().map(|v| v*v).sum::<f32>().sqrt();
        println!("기대 norm: {{exp_norm:.4}}, 실제 norm: {{out_norm:.4}}");

        for i in 0..{n_total} {{
            let diff = (EXPECT[i] - out[i]).abs();
            println!("  [{{i:2}}] PyTorch={{:.4}} RNGD={{:.4}} diff={{:.4}}", EXPECT[i], out[i], diff);
        }}

        let mut fail = 0;
        for i in 0..{n_total} {{
            let tol = (EXPECT[i].abs() * 0.10_f32).max(1.0_f32);
            if (EXPECT[i] - out[i]).abs() > tol {{ fail += 1; }}
        }}
        assert!(fail == 0, "타일링 불일치: {{fail}}/{n_total}");
        println!("✅ PASS: K={k_total}, N={n_total} 타일링 검증 성공");
    }}
}}
"""


def gen_tiled_linear_test(
        k_total: int,
        n_total: int,
        k_tile: int = 16,
        n_tile: int = 8,
) -> str:
    """gen_tiled_linear_host와 함께 쓸 PyTorch 기준값 생성 코드."""
    import torch
    torch.manual_seed(42)

    k_tiles = k_total // k_tile
    n_tiles = n_total // n_tile

    x = torch.randn(1, k_total).to(torch.bfloat16)
    w = torch.randn(n_total, k_total).to(torch.bfloat16)
    out = torch.nn.functional.linear(x, w)

    def fmt(lst):
        return ", ".join(f"{v:.8f}_f32" for v in lst)
    def tolist(t):
        return t.flatten().to(torch.float32).tolist()

    lines = [
        f"// 타일링 reference data: K={k_total}, N={n_total}",
        f"// K타일={k_tiles}, N타일={n_tiles}, 총 {k_tiles*n_tiles}회",
        f"pub const K_TOTAL: usize = {k_total};",
        f"pub const N_TOTAL: usize = {n_total};",
        f"pub const K_TILES: usize = {k_tiles};",
        f"pub const N_TILES: usize = {n_tiles};",
        f"const INPUT_F32: &[f32] = &[{fmt(tolist(x))}];",
    ]

    for n in range(n_tiles):
        for k in range(k_tiles):
            w_tile = w[n*n_tile:(n+1)*n_tile, k*k_tile:(k+1)*k_tile]
            lines.append(
                f"const W_N{n}_K{k}_F32: &[f32] = &[{fmt(tolist(w_tile))}];"
            )

    lines.append(f"const EXPECT_F32: &[f32] = &[{fmt(tolist(out))}];")
    lines.append(f"pub fn reference_input() -> Vec<f32> {{ INPUT_F32.to_vec() }}")
    for n in range(n_tiles):
        for k in range(k_tiles):
            lines.append(
                f"pub fn w_n{n}_k{k}() -> Vec<f32> {{ W_N{n}_K{k}_F32.to_vec() }}"
            )
    lines.append(f"pub fn reference_expected() -> Vec<f32> {{ EXPECT_F32.to_vec() }}")

    return "\n".join(lines)


if __name__ == "__main__" and False:
    # 타일링 PoC 테스트 (직접 실행 시)
    print(gen_tiled_linear_host(k_total=32, n_total=16))
    print("---")
    print(gen_tiled_linear_test(k_total=32, n_total=16))
