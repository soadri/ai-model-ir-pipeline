"""
명령줄 실행 — 웹 화면 없이 변환과 검증을 바로 실행합니다.

    python -m server.cli info
    python -m server.cli verify [--scale sim_small|sim_mid] [--json result.json]
    python -m server.cli convert my_model.py [--json result.json]

Docker 이미지에서:
    docker run --rm ai-model-ir-pipeline:dev python -m server.cli verify --scale sim_mid
"""
import argparse
import json
import sys
import time
from pathlib import Path

from rngd_pipeline import __version__, config
from rngd_pipeline.kernel_backend import backend_info


def _mark(ok) -> str:
    return "통과" if ok else "실패"


def _save(result: dict, path: str | None) -> None:
    if path:
        Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\n전체 결과를 저장했습니다: {path}")


def cmd_info(_args) -> int:
    print(json.dumps({"version": __version__, "kernel": backend_info(),
                      "scales": config.AVAILABLE_SCALES}, ensure_ascii=False, indent=2))
    return 0


def cmd_verify(args) -> int:
    from server.services import preset_verify

    if args.scale not in config.AVAILABLE_SCALES:
        print(f"사용할 수 없는 검증 규모입니다: {args.scale} (사용 가능: {', '.join(config.AVAILABLE_SCALES)})")
        return 2
    config.ensure_dirs()
    print(f"Llama-3.1 8B Decoder Layer 구조, 검증 규모 {args.scale} — 변환과 3단계 검증을 시작합니다.\n")
    started = time.time()
    result = preset_verify.run_all(scale=args.scale)

    l12, l3, ir = result.get("level12") or {}, result.get("level3") or {}, result.get("ir_detail") or {}
    print("\n" + "=" * 60)
    print(f"NPU dialect 연산    {ir.get('rngd_count', '-')}개, 변환 커버리지 {ir.get('coverage_pct', '-')}%")
    print(f"Level 1-2 IR 정합성  {l12.get('pass_count')}/{len(l12.get('checks', []))} {_mark(l12.get('all_passed'))}")
    for c in l12.get("checks", []):
        if not c.get("passed"):
            print(f"   실패: {c['name']} — {c.get('detail', '')}")
    print(f"Level 3 NPU 시뮬레이터 {l3.get('pass_count')}/{l3.get('total')} {_mark(l3.get('all_passed'))}")
    for c in l3.get("checks", []):
        print(f"   {'✓' if c.get('passed') else '✗'} {c['module']:<34} max_diff={c.get('max_diff')}")
    print("=" * 60 + f"\n소요 시간 {time.time() - started:.1f}초")
    _save(result, args.json)
    return 0 if l12.get("all_passed") and l3.get("all_passed") else 1


def cmd_convert(args) -> int:
    from server.services import custom_convert

    code = Path(args.model_file).read_text(encoding="utf-8")
    config.ensure_dirs()
    started = time.time()
    try:
        model, x = custom_convert.prepare(code)
    except custom_convert.InputError as e:
        print(f"입력 오류: {e}")
        return 2
    result = custom_convert.analyze(model, x, code, started)

    l12, l3 = result.get("level12") or {}, result.get("level3") or {}
    summary = l3.get("summary", {}) if isinstance(l3, dict) else {}
    print("=" * 60)
    print(f"변환 커버리지       {result['coverage_pct']}% (변환 {result['converted_count']}개, "
          f"미지원 {result['unsupported_count']}개)")
    for u in result.get("unsupported", []):
        print(f"   미지원: {u.get('label') or u.get('name')} — {u['reason']}")
    print(f"복합 패턴           {result.get('fusion') or '없음'}")
    print(f"Level 1-2 수치 비교  {_mark(l12.get('passed'))} (상대 오차 {l12.get('rel_err_pct')}%)")
    print(f"Level 3 NPU 시뮬레이터 {summary.get('passed', 0)}/{summary.get('tested', 0)} 통과")
    print("=" * 60 + f"\n소요 시간 {result['elapsed_s']}초")
    _save(result, args.json)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m server.cli", description="AI 모델 변환(표준 IR) 파이프라인")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="버전과 실행 환경")
    v = sub.add_parser("verify", help="기본 모델(Llama-3.1 8B Decoder Layer) 변환과 3단계 검증")
    v.add_argument("--scale", default="sim_small")
    v.add_argument("--json", help="전체 결과를 저장할 파일")
    c = sub.add_parser("convert", help="사용자 정의 모델 파일 변환과 검증 (model, x 변수 필요)")
    c.add_argument("model_file")
    c.add_argument("--json", help="전체 결과를 저장할 파일")
    args = ap.parse_args()
    return {"info": cmd_info, "verify": cmd_verify, "convert": cmd_convert}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
