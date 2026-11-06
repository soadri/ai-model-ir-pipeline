#!/usr/bin/env python3
"""
소개 페이지의 검증 기록(status.json)을 만듭니다.

  - 릴리즈: 각 버전의 Release에 첨부된 검증 결과(verify-*.json, convert-*.json)를 요약
  - Upstream: 고정 의존성 가용성과 Upstream 최신 버전 (upstream_watch.py와 같은 확인)

사용법: build_site_data.py <출력 파일>
환경변수: GITHUB_TOKEN(선택), GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_SERVER_URL (Actions가 제공)
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import upstream_watch as uw  # noqa: E402

REPO = uw.REPO
ASSETS = {"verify-sim_small.json": "sim_small", "verify-sim_mid.json": "sim_mid", "convert-swiglu.json": "custom"}


def download_json(url):
    req = urllib.request.Request(url, headers={**uw.UA, "Accept": "application/octet-stream",
                                               **({"Authorization": f"Bearer {uw.TOKEN}"} if uw.TOKEN else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def summarize_verify(d):
    l12, l3, ir = d.get("level12") or {}, d.get("level3") or {}, d.get("ir_detail") or {}
    return {
        "coverage": ir.get("coverage_pct"),
        "npu_ops": ir.get("rngd_count"),
        "l12": [l12.get("pass_count"), len(l12.get("checks", []))],
        "l3": [l3.get("pass_count"), l3.get("total")],
        "ok": bool(l12.get("all_passed") and l3.get("all_passed")),
        "modules": [{"name": c.get("module"), "rel": c.get("rel_err_pct"), "ok": c.get("passed")}
                    for c in l3.get("checks", [])],
    }


def summarize_convert(d):
    s = (d.get("level3") or {}).get("summary") or {}
    return {
        "coverage": d.get("coverage_pct"),
        "npu_ops": d.get("converted_count"),
        "l12": bool((d.get("level12") or {}).get("passed")),
        "l3": [s.get("passed", 0), s.get("tested", 0)],
        "ok": bool((d.get("level12") or {}).get("passed") and s.get("all_passed")),
    }


def releases():
    out = []
    for rel in uw.gh(f"/repos/{REPO}/releases?per_page=20"):
        if rel.get("draft") or rel.get("prerelease") or not rel["tag_name"].startswith("v"):
            continue
        item = {"tag": rel["tag_name"], "published": rel.get("published_at"), "url": rel["html_url"], "checks": {}}
        digest = re.search(r"sha256:[0-9a-f]{64}", rel.get("body") or "")
        if digest:
            item["digest"] = digest.group(0)
        for asset in rel.get("assets", []):
            kind = ASSETS.get(asset["name"])
            if not kind:
                continue
            try:
                data = download_json(asset["url"])
                item["checks"][kind] = summarize_convert(data) if kind == "custom" else summarize_verify(data)
            except Exception as e:  # noqa: BLE001
                item["checks"][kind] = {"error": str(e)}
        out.append(item)
    return out


def upstream():
    pinned = uw.check_wheels() + [uw.check_crate_pinned()]
    latest = uw.latest_versions()
    pin_of = {"PyTorch (안정)": uw.PINNED["torch"], "torch-mlir (개발 wheel)": uw.PINNED["torch-mlir"],
              "furiosa-opt-std": uw.PINNED["furiosa-opt-std"], "cargo-furiosa-opt": uw.PINNED["furiosa-opt-std"]}
    return {
        "pinned": pinned,
        "latest": [{"name": k, "pinned": pin_of.get(k), "latest": v} for k, v in latest.items()],
        "ok": all(r["ok"] for r in pinned),
    }


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "status.json")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run = os.environ.get("GITHUB_RUN_ID")
    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_url": f"{server}/{REPO}/actions/runs/{run}" if run else None,
        "repository": f"{server}/{REPO}",
        "releases": releases(),
        "upstream": upstream(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}: 릴리즈 {len(status['releases'])}개, 고정 의존성 {'정상' if status['upstream']['ok'] else '문제 있음'}")


if __name__ == "__main__":
    main()
