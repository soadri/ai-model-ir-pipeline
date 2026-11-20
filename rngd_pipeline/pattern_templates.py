"""
예시 기반 패턴 식별: 템플릿 저장소와 규칙 기반과의 비교

템플릿은 「예시 모델 + 라벨을 붙인 노드」로 만든다(pattern_matcher.extract_pattern_from_example).
어떤 예시로 만들었는지는 EXAMPLES에 적어 두고, 만든 결과는 registry/pattern_templates.json에 저장한다.

    python -m rngd_pipeline.pattern_templates            # 템플릿 목록
    python -m rngd_pipeline.pattern_templates --check    # 예시에서 다시 만들어 저장된 템플릿과 같은지 확인
    python -m rngd_pipeline.pattern_templates --rebuild  # 예시에서 다시 만들어 저장

라벨 노드의 연산이 EXAMPLES에 적은 연산과 다르면(파이프라인이 바뀌어 노드 번호가 달라진 경우 등)
다시 만들지 않고 멈춘다. 이때는 라벨을 다시 확인해 EXAMPLES를 고친다.

Attention 템플릿은 두지 않는다. 노드 20개가 넘는 큰 패턴을 템플릿 하나로 찾는 방식은 시험한 모델에서
한 번도 맞지 않았다(하위 패턴을 조합하는 계층적 매칭이 필요). Attention은 규칙 기반 결과만 쓴다.
"""
import json
import sys
from pathlib import Path

from . import pattern_matcher as pm

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "registry" / "pattern_templates.json"

# 예시: (템플릿 이름, 종류, 변형, 예시 모델, 라벨 노드, 라벨 노드의 연산, 하위 패턴)
EXAMPLES = [
    ("RMSNORM", "RMSNORM", "", "llama_sim_small", [0, 1, 2, 3, 4, 5, 6],
     ["elementwise(pow2)", "reduce(sum)", "elementwise(div)", "elementwise(add)", "elementwise(rsqrt)", "elementwise(mul)", "elementwise(mul)"], []),
    ("SOFTMAX", "SOFTMAX", "", "llama_sim_small", [26, 27, 28, 29, 30],
     ["reduce(max)", "elementwise(sub)", "elementwise(exp)", "reduce(sum)", "elementwise(div)"], []),
    ("SILU", "SILU", "", "llama_sim_small", [43, 44], ["elementwise(sigmoid)", "elementwise(mul)"], []),
    ("RESIDUAL", "RESIDUAL", "", "llama_sim_small", [34], ["elementwise(add)"], []),
    ("FFN", "FFN", "SwiGLU", "llama_sim_small", [42, 43, 44, 45, 46, 47],
     ["gemm", "elementwise(sigmoid)", "elementwise(mul)", "gemm", "elementwise(mul)", "gemm"], ["SILU"]),
    ("GELU", "GELU", "", "gelu_ffn", [1], ["elementwise(gelu)"], []),
    ("RELU", "RELU", "", "reglu_ffn", [1], ["elementwise(relu)"], []),
    ("FFN_GELU", "FFN", "GELU", "gelu_ffn", [0, 1, 2], ["gemm", "elementwise(gelu)", "gemm"], []),
    ("FFN_GEGLU", "FFN", "GeGLU", "geglu_ffn", [0, 1, 2, 3, 4],
     ["gemm", "elementwise(gelu)", "gemm", "elementwise(mul)", "gemm"], []),
    ("FFN_REGLU", "FFN", "ReGLU", "reglu_ffn", [0, 1, 2, 3, 4],
     ["gemm", "elementwise(relu)", "gemm", "elementwise(mul)", "gemm"], []),
]

MODEL_NOTES = {
    "llama_sim_small": "Llama-3.1 8B Decoder Layer 구조, 기본 규모(dim 16) — 검증용 모델과 같은 구성",
    "gelu_ffn": "Linear(64→128) → nn.GELU → Linear(128→64) (GPT-2·BERT 식)",
    "geglu_ffn": "gate·up Linear(64→128), GELU(gate)·up → Linear(128→64) (Gemma 식)",
    "reglu_ffn": "gate·up Linear(64→128), ReLU(gate)·up → Linear(128→64) (PaLM 식)",
}

# 모양(add 하나)만으로는 구분되지 않아 맥락 조건(Skip Connection)으로만 찾는 패턴
CONTEXT_PATTERNS = {"RESIDUAL"}

# 예시 기반 식별에 템플릿이 없는 패턴 (규칙 기반 결과만 있음)
RULE_ONLY = {"ATTENTION"}


def _label(n) -> str:
    return n.op_type.replace("rngd.", "") + (f"({n.op_attr})" if n.op_attr else "")


def example_nodes(model_key: str):
    """예시 모델의 데이터 흐름 그래프 (NPU 특화 IR 변환 후)"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_mlir
    from torch_mlir.ir import Context, Module

    from .e2e_pipeline import extract_dataflow, rewrite_to_rngd

    class _FFN(nn.Module):
        def __init__(self, act, gated, d=64, h=128):
            super().__init__()
            self.act, self.gated = act, gated
            self.gate = nn.Linear(d, h, bias=False)
            self.up = nn.Linear(d, h, bias=False)
            self.down = nn.Linear(h, d, bias=False)

        def forward(self, x):
            if self.gated:
                return self.down(self.act(self.gate(x)) * self.up(x))
            return self.down(self.act(self.up(x)))

    if model_key == "llama_sim_small":
        from .models import LlamaDecoderLayer
        from .npu_direct_verify import VERIFY_PROFILES
        c = VERIFY_PROFILES["sim_small"]
        torch.manual_seed(42)
        model = LlamaDecoderLayer(c["dim"], c["hidden"], c["n_heads"], c["n_kv_heads"], c["head_dim"],
                                  c["seq_len"], c["rms_eps"], always_repeat_kv=True)
        x = torch.randn(1, c["seq_len"], c["dim"])
    else:
        act, gated = {"gelu_ffn": (F.gelu, False), "geglu_ffn": (F.gelu, True), "reglu_ffn": (F.relu, True)}[model_key]
        torch.manual_seed(0)
        model, x = _FFN(act, gated), torch.randn(1, 8, 64)
    module = Module.parse(str(torch_mlir.compile(model.eval(), (x,), output_type="linalg-on-tensors",
                                                 use_make_fx=True)), Context())
    rewrite_to_rngd(module)
    return extract_dataflow(module)


def build_library() -> "pm.PatternLibrary":
    """EXAMPLES에서 템플릿을 만든다. 라벨 노드의 연산이 기록과 다르면 ValueError."""
    lib = pm.PatternLibrary()
    cache = {}
    for name, kind, variant, model_key, labels, ops, subs in EXAMPLES:
        if model_key not in cache:
            cache[model_key] = example_nodes(model_key)
        nodes = cache[model_key]
        by_idx = {n.idx: n for n in nodes}
        actual = [_label(by_idx[i]) if i in by_idx else "없음" for i in labels]
        if actual != ops:
            raise ValueError(f"{name}: 예시 {model_key}의 라벨 노드 {labels}의 연산이 기록과 다릅니다.\n"
                             f"  기록: {ops}\n  실제: {actual}")
        p = pm.extract_pattern_from_example(nodes, labels, name, description=MODEL_NOTES[model_key])
        p.kind, p.variant, p.sub_patterns = kind, variant, subs
        p.match_mode = "context" if name in CONTEXT_PATTERNS else ""
        p.source = {"model": model_key, "labels": labels, "ops": ops}
        lib.add_pattern(p)
    return lib


def load_library(path: Path = TEMPLATE_PATH) -> "pm.PatternLibrary":
    lib = pm.PatternLibrary()
    lib.load(str(path))
    return lib


def identify(nodes, lib=None) -> list[dict]:
    """예시 기반 식별 결과: [{kind, variant, template, nodes, similarity}]"""
    lib = lib or load_library()
    out = []
    for m in lib.find_all(nodes, threshold=0.7, strict=True):
        p = lib.patterns[m.pattern_name]
        out.append({"kind": p.kind or p.name, "variant": p.variant, "template": p.name,
                    "nodes": sorted(m.matched_indices), "similarity": round(m.similarity, 3)})
    return out


def compare(rule_groups, matches) -> dict:
    """규칙 기반 그룹과 예시 기반 매칭을 패턴 인스턴스 단위로 비교 (노드 겹침 |∩|/|∪| ≥ 0.8이면 일치)"""
    known = {m["kind"] for m in matches} | {p for p in _template_kinds()}
    rule = [(g.name, set(g.node_indices)) for g in rule_groups if g.name in known]
    rule_only = sorted({g.name for g in rule_groups if g.name not in known})
    found = [(m["kind"], set(m["nodes"])) for m in matches]
    used, agree, missed = set(), 0, []
    for name, s in rule:
        best, bi = 0.0, None
        for i, (k, f) in enumerate(found):
            if k == name and i not in used:
                j = len(s & f) / len(s | f)
                if j > best:
                    best, bi = j, i
        if best >= 0.8:
            agree += 1
            used.add(bi)
        else:
            missed.append({"pattern": name, "nodes": sorted(s), "best_overlap": round(best, 2)})
    extra = [{"pattern": k, "nodes": sorted(f)} for i, (k, f) in enumerate(found) if i not in used]
    return {"agree": agree, "total": len(rule), "missed": missed, "extra": extra, "rule_only": rule_only}


def _template_kinds():
    try:
        return {(p.kind or p.name) for p in load_library().patterns.values()}
    except FileNotFoundError:
        return set()


def report(nodes, rule_groups) -> dict:
    """결과 화면·명령줄에 붙이는 요약. 실패해도 기존 결과에 영향이 없도록 오류는 문자열로 돌려준다."""
    try:
        matches = identify(nodes)
        counts: dict[str, int] = {}
        for m in matches:
            counts[m["kind"]] = counts.get(m["kind"], 0) + 1
        return {"engine": "예시 기반 (그래프 템플릿 매칭)", "counts": counts, "matches": matches,
                "comparison": compare(rule_groups, matches)}
    except Exception as e:  # noqa: BLE001
        return {"engine": "예시 기반 (그래프 템플릿 매칭)", "error": f"{type(e).__name__}: {e}"}


def _main(argv) -> int:
    if "--rebuild" in argv or "--check" in argv:
        lib = build_library()
        new = {k: p.to_dict() for k, p in lib.patterns.items()}
        if "--rebuild" in argv:
            TEMPLATE_PATH.write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"템플릿 {len(new)}개를 저장했습니다: {TEMPLATE_PATH}")
            return 0
        old = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        if old == new:
            print(f"저장된 템플릿 {len(new)}개가 예시에서 다시 만든 결과와 같습니다.")
            return 0
        diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        print(f"저장된 템플릿이 예시에서 다시 만든 결과와 다릅니다: {diff}")
        return 1
    lib = load_library()
    print(f"템플릿 {len(lib.patterns)}개 ({TEMPLATE_PATH.name})")
    for p in lib.patterns.values():
        kind = (p.kind or p.name) + (f" ({p.variant})" if p.variant else "")
        print(f"  {p.name:<10} {kind:<16} 노드 {p.size:>2}개  예시 {p.source.get('model', '-')} {p.source.get('labels', '')}")
    print(f"템플릿 없음(규칙 기반만): {', '.join(sorted(RULE_ONLY))}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
