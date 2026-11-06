#!/usr/bin/env python3
"""
Upstream 변화 추적 — 파이프라인이 고정해서 쓰는 의존성과 상류 생태계의 변화를 확인합니다.

  1) 고정 의존성을 지금도 받을 수 있는가
     - 재구성 wheel(build-deps-1)의 존재와 SHA-256
     - furiosa-opt-std 0.6.0 (crates.io, 철회 여부)
  2) 상류에 새 버전이 나왔는가
     - PyTorch 안정 버전, torch-mlir 개발 wheel, furiosa-opt-std / cargo-furiosa-opt

결과는 Actions 요약에 기록하고, 직전 확인과 달라졌으면 `upstream` 라벨의 Issue를 만들거나 갱신합니다.
환경변수: GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_STEP_SUMMARY (Actions가 제공)
"""
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY", "soadri/ai-model-ir-pipeline")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
UA = {"User-Agent": f"{REPO} upstream-watch (github.com/{REPO})"}
LABEL = "upstream"
TITLE = "Upstream 변화 감지"
MARK = "<!-- upstream-fingerprint:"

PINNED = {"torch": "2.3.0.dev20240122+cpu", "torch-mlir": "20240127.1096", "furiosa-opt-std": "0.6.0"}


def get(url, headers=None, raw=False):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        return body.decode("utf-8", "replace") if raw else json.loads(body)


def gh(path, method="GET", data=None):
    req = urllib.request.Request(
        f"https://api.github.com{path}", method=method,
        data=json.dumps(data).encode() if data is not None else None,
        headers={**UA, "Accept": "application/vnd.github+json",
                 **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        return json.loads(body) if body else {}


# ── 1) 고정 의존성 ───────────────────────────────────────────────────

def check_wheels():
    expected = {}
    for line in Path("docker/wheels.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            expected[name] = digest
    try:
        rel = gh(f"/repos/{REPO}/releases/tags/build-deps-1")
    except urllib.error.HTTPError as e:
        return [{"name": n, "ok": False, "note": f"Release를 찾을 수 없음 ({e.code})"} for n in expected]
    assets = {a["name"]: a for a in rel.get("assets", [])}
    rows = []
    for name, digest in expected.items():
        a = assets.get(name)
        if not a:
            rows.append({"name": name, "ok": False, "note": "파일 없음"})
            continue
        got = (a.get("digest") or "").removeprefix("sha256:")
        if got and got != digest:
            rows.append({"name": name, "ok": False, "note": f"체크섬 불일치 ({got[:12]}…)"})
        else:
            rows.append({"name": name, "ok": True, "note": "체크섬 일치" if got else "존재 확인"})
    return rows


def crate(name):
    return get(f"https://crates.io/api/v1/crates/{name}")


def check_crate_pinned():
    data = crate("furiosa-opt-std")
    versions = {v["num"]: v for v in data["versions"]}
    v = versions.get(PINNED["furiosa-opt-std"])
    if not v:
        return {"name": "furiosa-opt-std 0.6.0", "ok": False, "note": "crates.io에서 찾을 수 없음"}
    if v.get("yanked"):
        return {"name": "furiosa-opt-std 0.6.0", "ok": False, "note": "철회(yanked)됨"}
    return {"name": "furiosa-opt-std 0.6.0", "ok": True, "note": "crates.io에서 받을 수 있음"}


# ── 2) 상류 최신 버전 ───────────────────────────────────────────────

def latest_versions():
    out = {}
    try:
        out["PyTorch (안정)"] = get("https://pypi.org/pypi/torch/json")["info"]["version"]
    except Exception as e:  # noqa: BLE001
        out["PyTorch (안정)"] = f"확인 실패: {e}"
    try:
        html = get("https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels", raw=True)
        dates = re.findall(r"torch_mlir-(\d{8})\.(\d+)-", html)
        out["torch-mlir (개발 wheel)"] = max(f"{d}.{n}" for d, n in dates) if dates else "확인 실패"
    except Exception as e:  # noqa: BLE001
        out["torch-mlir (개발 wheel)"] = f"확인 실패: {e}"
    for name in ("furiosa-opt-std", "cargo-furiosa-opt"):
        try:
            out[name] = crate(name)["crate"]["max_version"]
        except Exception as e:  # noqa: BLE001
            out[name] = f"확인 실패: {e}"
    return out


# ── 보고서와 Issue ──────────────────────────────────────────────────

def build_report(pinned_rows, latest):
    lines = ["## 고정 의존성 가용성", "", "| 항목 | 상태 | 비고 |", "|:--|:--:|:--|"]
    for r in pinned_rows:
        lines.append(f"| `{r['name']}` | {'✅' if r['ok'] else '❌'} | {r['note']} |")
    lines += ["", "## 상류 최신 버전", "", "| 구성 요소 | 파이프라인 고정 버전 | 상류 최신 |", "|:--|:--|:--|"]
    pin_of = {"PyTorch (안정)": PINNED["torch"], "torch-mlir (개발 wheel)": PINNED["torch-mlir"],
              "furiosa-opt-std": PINNED["furiosa-opt-std"], "cargo-furiosa-opt": PINNED["furiosa-opt-std"]}
    for k, v in latest.items():
        lines.append(f"| {k} | {pin_of.get(k, '–')} | {v} |")
    lines += ["", "고정 의존성에 ❌가 있으면 체험 이미지를 다시 빌드할 수 없는 상태입니다. "
              "상류 최신 버전이 바뀐 경우에는 같은 검증 기준(Level 1-2, Level 3)으로 호환성을 확인한 뒤 고정 버전 갱신을 검토합니다."]
    return "\n".join(lines)


def fingerprint(pinned_rows, latest):
    key = json.dumps({"pinned": [(r["name"], r["ok"], r["note"]) for r in pinned_rows],
                      "latest": {k: v for k, v in latest.items() if not str(v).startswith("확인 실패")}},
                     sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def sync_issue(report, fp, problem):
    if not TOKEN:
        print("GITHUB_TOKEN이 없어 Issue 갱신을 건너뜁니다.")
        return
    try:
        gh(f"/repos/{REPO}/labels", "POST", {"name": LABEL, "color": "5B4BC4", "description": "Upstream 생태계 변화 추적"})
    except urllib.error.HTTPError:
        pass  # 이미 있음
    issues = gh(f"/repos/{REPO}/issues?state=open&labels={LABEL}&per_page=10")
    body = f"{report}\n\n{MARK}{fp} -->"
    if issues:
        issue = issues[0]
        if MARK + fp in (issue.get("body") or ""):
            print(f"변화 없음 (Issue #{issue['number']} 유지)")
            return
        gh(f"/repos/{REPO}/issues/{issue['number']}", "PATCH", {"body": body})
        gh(f"/repos/{REPO}/issues/{issue['number']}/comments", "POST",
           {"body": "직전 확인 이후 Upstream 상태가 바뀌어 본문을 갱신했습니다." + (" **고정 의존성에 문제가 있습니다.**" if problem else "")})
        print(f"Issue #{issue['number']} 갱신")
    else:
        created = gh(f"/repos/{REPO}/issues", "POST", {"title": TITLE, "body": body, "labels": [LABEL]})
        print(f"Issue #{created['number']} 생성")


def main():
    pinned_rows = check_wheels() + [check_crate_pinned()]
    latest = latest_versions()
    report = build_report(pinned_rows, latest)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("# Upstream 변화 추적\n\n" + report + "\n")
    problem = not all(r["ok"] for r in pinned_rows)
    sync_issue(report, fingerprint(pinned_rows, latest), problem)
    sys.exit(1 if problem else 0)


if __name__ == "__main__":
    main()
