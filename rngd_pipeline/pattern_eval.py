"""
예시 기반 패턴 식별 평가: 규칙 기반 식별과의 일치율

    python -m rngd_pipeline.pattern_eval

평가 모델: Llama-3.1 8B Decoder Layer 구조 2개 규모(기본, 중간 GQA)와 대시보드 사용자 모델 예시 전체.
패턴 인스턴스마다 규칙 기반 그룹과 예시 기반 매칭의 노드 겹침(|∩|/|∪|)이 0.8 이상이면 일치로 본다.
템플릿이 없는 패턴(Attention)은 비교에서 빼고 따로 표시한다.

CI 기준: 예시 기반만 찾은 패턴(오탐)이 없고, 일치율이 MIN_AGREEMENT 이상이면 통과(종료 코드 0).
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

MIN_AGREEMENT = 1.0   # 현재 평가 모델에서 규칙 기반과 모두 일치 (기준을 낮추면 회귀를 놓친다)
EXAMPLES_JS = Path(__file__).resolve().parent.parent / "web" / "js" / "examples.js"


def _graphs():
    import torch
    import torch_mlir
    from torch_mlir.ir import Context, Module

    from .e2e_pipeline import extract_dataflow, rewrite_to_rngd
    from .models import LlamaDecoderLayer
    from .npu_direct_verify import VERIFY_PROFILES

    def nodes_of(model, x):
        module = Module.parse(str(torch_mlir.compile(model.eval(), (x,), output_type="linalg-on-tensors",
                                                     use_make_fx=True)), Context())
        rewrite_to_rngd(module)
        return extract_dataflow(module)

    for scale, title in (("sim_small", "Llama Decoder Layer · 기본 규모"), ("sim_mid", "Llama Decoder Layer · 중간 규모 GQA")):
        c = VERIFY_PROFILES[scale]
        torch.manual_seed(42)
        m = LlamaDecoderLayer(c["dim"], c["hidden"], c["n_heads"], c["n_kv_heads"], c["head_dim"], c["seq_len"],
                              c["rms_eps"], always_repeat_kv=True)
        yield title, nodes_of(m, torch.randn(1, c["seq_len"], c["dim"]))

    text = EXAMPLES_JS.read_text(encoding="utf-8")
    examples = json.loads(re.sub(r",\s*\]\s*;?\s*$", "]", text[text.index("["):].strip()))
    for e in examples:
        ns: dict = {}
        exec(e["code"], ns)  # noqa: S102 — 저장소에 들어 있는 대시보드 예시 코드
        yield f"예시 · {e['title']}", nodes_of(ns["model"], ns["x"])


def run(verbose: bool = True) -> dict:
    from .e2e_pipeline import detect_fusion_patterns
    from .pattern_templates import compare, identify, load_library

    lib = load_library()
    tot, by_pattern, rule_only = Counter(), {}, Counter()
    rows, slowest = [], 0.0
    for title, nodes in _graphs():
        rule = detect_fusion_patterns(nodes)
        t = time.time()
        matches = identify(nodes, lib)
        slowest = max(slowest, (time.time() - t) * 1000)
        c = compare(rule, matches)
        tot["agree"] += c["agree"]
        tot["total"] += c["total"]
        tot["extra"] += len(c["extra"])
        for name in c["rule_only"]:
            rule_only[name] += 1
        for g in rule:
            if g.name in c["rule_only"]:
                continue
            by_pattern.setdefault(g.name, Counter())["total"] += 1
        for miss in c["missed"]:
            by_pattern[miss["pattern"]]["missed"] += 1
        rows.append((title, len(nodes), c))
    rate = tot["agree"] / tot["total"] if tot["total"] else 1.0
    result = {"agree": tot["agree"], "total": tot["total"], "rate": round(rate, 4), "extra": tot["extra"],
              "by_pattern": {k: {"agree": v["total"] - v["missed"], "total": v["total"]} for k, v in sorted(by_pattern.items())},
              "rule_only": dict(rule_only), "slowest_ms": round(slowest, 1),
              "passed": tot["extra"] == 0 and rate >= MIN_AGREEMENT}
    if verbose:
        print("예시 기반 패턴 식별 — 규칙 기반과의 일치")
        for title, n, c in rows:
            mark = "✓" if c["agree"] == c["total"] and not c["extra"] else "✗"
            note = ""
            if c["missed"]:
                note += " 놓침 " + ", ".join(m["pattern"] for m in c["missed"])
            if c["extra"]:
                note += " 예시 기반만 " + ", ".join(x["pattern"] for x in c["extra"])
            print(f"  {mark} {title:<40} 노드 {n:>2}개  일치 {c['agree']}/{c['total']}{note}")
        print(f"합계: 일치 {result['agree']}/{result['total']} ({rate * 100:.1f}%), 예시 기반만 찾은 패턴 {result['extra']}개")
        print("패턴별:", ", ".join(f"{k} {v['agree']}/{v['total']}" for k, v in result["by_pattern"].items()))
        if rule_only:
            print("템플릿 없음(규칙 기반만):", ", ".join(f"{k} {v}개" for k, v in rule_only.items()))
        print(f"매칭 시간: 모델당 최대 {result['slowest_ms']}ms")
        print("결과:", "통과" if result["passed"] else f"미통과 (기준: 예시 기반만 찾은 패턴 0개, 일치율 {MIN_AGREEMENT * 100:.0f}% 이상)")
    return result


if __name__ == "__main__":
    sys.exit(0 if run()["passed"] else 1)
