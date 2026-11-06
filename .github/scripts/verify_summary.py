#!/usr/bin/env python3
"""CLI 검증 결과(JSON)를 GitHub Actions 요약(마크다운)으로 정리합니다.

사용법: verify_summary.py <결과.json> <제목>   → 표준 출력으로 마크다운
"""
import json
import sys


def mark(ok):
    return "✅" if ok else "❌"


def main():
    path, title = sys.argv[1], sys.argv[2]
    d = json.load(open(path, encoding="utf-8"))
    l12, l3, ir = d.get("level12") or {}, d.get("level3") or {}, d.get("ir_detail") or {}
    out = [f"### {title}", "",
           "| 항목 | 결과 |", "|:--|:--|",
           f"| 변환 커버리지 | {ir.get('coverage_pct', '–')}% |",
           f"| NPU dialect 연산 | {ir.get('rngd_count', '–')}개 |",
           f"| Level 1-2 IR 정합성 | {mark(l12.get('all_passed'))} {l12.get('pass_count')}/{len(l12.get('checks', []))} |",
           f"| Level 3 NPU 시뮬레이터 | {mark(l3.get('all_passed'))} {l3.get('pass_count')}/{l3.get('total')} |",
           "", "| Level 3 모듈 | 상대 오차 | 판정 |", "|:--|--:|:--:|"]
    for c in l3.get("checks", []):
        rel = c.get("rel_err_pct")
        out.append(f"| {c.get('module')} | {f'{rel:.4g}%' if rel is not None else '–'} | {mark(c.get('passed'))} |")
    print("\n".join(out))
    ok = l12.get("all_passed") and l3.get("all_passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
