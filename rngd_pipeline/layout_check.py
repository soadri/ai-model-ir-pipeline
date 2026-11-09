"""
데이터 배치 연산(전치·확장·복사)의 의미 점검 — Level 1-2 보조 검사

NPU 특화 IR 재작성이 '입력을 그대로 내보내는' linalg.generic(전치, 확장, 복사)을
올바른 연산(rngd.transpose / rngd.broadcast / 행렬 곱의 transpose_rhs)으로 바꾸고,
축 순서(perm)와 확장 축(broadcast_dims)을 정확히 기록하는지 확인한다.

방법: 변환 전 linalg IR의 해당 연산에 꼬리표(loc("probeN"))를 달고 재작성을 실행한 뒤,
      indexing_maps로 판정한 실제 성격과 재작성 결과를 대조한다.

  python -m rngd_pipeline.layout_check        # 문제 연산만 출력, 문제가 있으면 종료 코드 1
  python -m rngd_pipeline.layout_check -v     # 모든 연산 출력

배경: v0.1.1까지는 shape만 보고 판별해, 축 크기가 같은 전치가 broadcast로,
      GQA의 키·값 헤드 확장이 transpose로 표시되었다 (v0.1.2에서 수정, CHANGELOG 참고).
"""
import argparse
import re
import sys

import torch
import torch_mlir
from torch_mlir.ir import Context, Module

from .e2e_pipeline import rewrite_to_rngd
from .models import LlamaDecoderLayer

CONFIGS = {
    "sim_small (seq 8, head 16)": dict(dim=16, hidden=8, n_heads=1, n_kv_heads=1, head_dim=16, seq=8),
    "sim_mid (seq 8, head 8, 정사각형)": dict(dim=64, hidden=128, n_heads=8, n_kv_heads=2, head_dim=8, seq=8),
    "비정사각형 (seq 12, head 16)": dict(dim=64, hidden=128, n_heads=4, n_kv_heads=2, head_dim=16, seq=12),
}


# ── 1) 꼬리표 달기 ─────────────────────────────────────────────────────
def tag_identity_generics(ir: str):
    """body가 'linalg.yield %in' 하나뿐인 linalg.generic의 닫는 줄에 loc("probeN")을 붙인다."""
    lines = ir.split("\n")
    maps = {m.group(1): m.group(2) for m in (re.match(r"(#map\d*) = affine_map<(.*)>$", l.strip()) for l in lines) if m}
    probes, i = [], 0
    while i < len(lines):
        line = lines[i]
        if "= linalg.generic" in line:
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].strip().startswith("} -> "):
                body.append(lines[j].strip())
                j += 1
            ops = [b for b in body if b and not b.startswith("^bb")]
            m_yield = re.fullmatch(r"linalg\.yield (%\w+) : \S+", ops[0]) if len(ops) == 1 else None
            m_arg = re.search(r"\^bb0\((%\w+):", " ".join(body))
            if m_yield and m_arg and m_yield.group(1) == m_arg.group(1) and j < len(lines):
                n = len(probes)
                im = re.search(r"indexing_maps = \[([^\]]*)\]", line)
                ms = [maps.get(x.strip(), x.strip()) for x in im.group(1).split(",")] if im else []
                ins = re.search(r"ins\(%[\w#:]+ : (tensor<[^>]*>)\)", line)
                outs = re.search(r"\} -> (tensor<[^>]*>)", lines[j])
                probes.append({"id": f"probe{n}", "maps": ms,
                               "in": ins.group(1) if ins else "?", "out": outs.group(1) if outs else "?",
                               "line": line.strip()[:120]})
                lines[j] = lines[j] + f' loc("probe{n}")'
            i = j
        i += 1
    return "\n".join(lines), probes


# ── 2) 실제로 무엇인지 판정 ────────────────────────────────────────────
def parse_map(s):
    m = re.match(r"\(([^)]*)\)\s*->\s*\(([^)]*)\)", s)
    dims = [d.strip() for d in m.group(1).split(",") if d.strip()]
    res = [r.strip() for r in m.group(2).split(",") if r.strip()]
    return dims, res


def truth(p):
    """두 indexing map(입력 I, 출력 O)의 관계로 판정. 출력 축 j는 입력 축 perm[j]에서 온다."""
    if len(p["maps"]) < 2:
        return "판정 불가", None
    dims, rin = parse_map(p["maps"][0])
    _, rout = parse_map(p["maps"][-1])
    pure = lambda r: len(r) == len(dims) and all(x in dims for x in r) and len(set(r)) == len(r)
    if not pure(rout):
        return "판정 불가", None
    if not pure(rin):
        return "확장", None          # 입력 쪽에 상수(0) 축이 있거나 축이 적음 = 크기 1 축을 늘림
    perm = [rin.index(d) for d in rout]
    if perm == list(range(len(perm))):
        return "복사", None
    return "전치", perm


def shape(t):
    return [int(x) for x in re.findall(r"(\d+)x", t + "x")] if t != "?" else []


# ── 3·4) 재작성 후 추적 ────────────────────────────────────────────────
def outcome(probes, after_ir):
    alias = dict(re.findall(r'^(#loc\d+) = loc\("(probe\d+)"\)', after_ir, re.M))
    found = {}
    for line in after_ir.split("\n"):
        if line.strip().startswith("#loc"):      # 위치 정보 정의 줄은 연산이 아님
            continue
        m = re.search(r'loc\((#loc\d+|"probe\d+")\)\s*$', line.strip())
        if not m:
            continue
        key = m.group(1)
        pid = alias.get(key) or key.strip('"')
        if not pid.startswith("probe"):
            continue
        op = re.search(r'"?((?:rngd|linalg|tensor)\.[\w.]+)"?', line)
        attrs = re.search(r"\{([^}]*)\}", line)
        found[pid] = (op.group(1) if op else "?", attrs.group(1) if attrs else "")
    return found


def judge(kind, perm, res, sin, sout):
    if res is None:
        if kind == "전치" and perm is not None and perm[:-2] == list(range(len(perm) - 2)) and perm[-2:] == [len(perm) - 1, len(perm) - 2]:
            return "흡수", "마지막 두 축 전치 → transpose_rhs 결합 가능 (해당 행렬 곱의 속성 확인 필요)"
        return "흡수", f"⚠ 실제는 {kind}{perm or ''}인데 행렬 곱 체인에 흡수됨 → 정보가 사라졌는지 확인 필요"
    name, attrs = res
    if name == "?":
        return "추적 실패", "재작성 후 IR에서 연산 이름을 찾지 못함 (스크립트 확인 필요)"
    rec_perm = re.search(r'perm = "([^"]*)"', attrs)
    rec_bd = re.search(r'broadcast_dims = "([^"]*)"', attrs)
    if kind == "전치":
        if name == "rngd.transpose":
            if rec_perm is None:
                return name, "이름 일치. 단 축 순서(perm)가 IR에 기록되지 않음"
            if rec_perm.group(1) == str(perm):
                return name, f"✓ 이름·축 순서 일치 (perm={rec_perm.group(1)})"
            return name, f"✗ 기록된 축 순서 {rec_perm.group(1)}가 실제 {perm}와 다름"
        return name, f"✗ 전치(perm={perm})가 {name.split('.')[-1]}로 잘못 표시됨, 축 순서 정보 손실"
    if kind == "확장":
        if name != "rngd.broadcast":
            return name, f"✗ 확장이 {name}로 표시됨"
        return name, ("이름 일치 (확장 축은 기록되지 않음)" if rec_bd is None else f"✓ 이름 일치, 확장 축 {rec_bd.group(1)} 기록")
    if kind == "복사":
        return name, ("이름 부정확 (실제는 단순 복사)" if name == "rngd.broadcast" else f"✗ 복사가 {name}로 표시됨")
    return name, kind


def run(name, cfg, verbose=False):
    torch.manual_seed(42)
    model = LlamaDecoderLayer(cfg["dim"], cfg["hidden"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
                              cfg["seq"], always_repeat_kv=True).eval()
    x = torch.randn(1, cfg["seq"], cfg["dim"])
    compiled = torch_mlir.compile(model, (x,), output_type="linalg-on-tensors", use_make_fx=True)
    ir = str(compiled)
    tagged, probes = tag_identity_generics(ir)
    ctx = Context()
    module = Module.parse(tagged, ctx)
    rewrite_to_rngd(module)
    after = module.operation.get_asm(enable_debug_info=True)
    res = outcome(probes, after)

    print("=" * 110)
    print(f"{name}: 입력을 그대로 내보내는 linalg.generic {len(probes)}개")
    print("-" * 110)
    bad = 0
    for p in probes:
        kind, perm = truth(p)
        r = res.get(p["id"])
        got, verdict = judge(kind, perm, r, shape(p["in"]), shape(p["out"]))
        bad += verdict.startswith(("✗", "⚠", "이름 일치", "이름 부정확"))
        if verbose or verdict.startswith(("✗", "⚠", "이름", "재작성")):
            print(f"{p['id']:>8}  {p['in']:>28} → {p['out']:<28}  실제: {kind}{'' if perm is None else str(perm):<14}  결과: {got:<16} {verdict}")
    gemms = re.findall(r'"rngd\.(batch_gemm|gemm)"\(([^)]*)\)\s*(\{[^}]*\})?', after)
    tr = sum(1 for g in gemms if g[2] and "transpose_rhs" in g[2])
    print("-" * 110)
    print(f"행렬 곱: {len(gemms)}개 (transpose_rhs 붙은 것 {tr}개) / 확인 필요 {bad}건")
    return bad


def main():
    ap = argparse.ArgumentParser(description="데이터 배치 연산의 의미 점검")
    ap.add_argument("-v", "--verbose", action="store_true", help="모든 연산 출력")
    a = ap.parse_args()
    total = sum(run(n, c, a.verbose) for n, c in CONFIGS.items())
    print("=" * 110)
    print(f"전체 확인 필요 {total}건")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
