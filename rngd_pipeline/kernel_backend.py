"""
커널 실행 백엔드 — Level 3 검증에서 Rust 커널을 시뮬레이터로 실행합니다.

    from rngd_pipeline.kernel_backend import run_kernel

    passed, log, stats = run_kernel("pilot_e2e_add")

    # 입력 데이터를 주입해서 실행 (실행 후 원래 파일로 자동 복원)
    passed, log, stats = run_kernel(
        "pilot_golden_ffn_gate",
        inputs={"golden_ffn_gate_a.bin": a_bytes, ...},
    )

백엔드는 RNGD_KERNEL_BACKEND 환경변수로 선택합니다 (config.py 참고).
  local  : 현재 환경에서 실행 (체험 이미지 기본값). 빌드 단계에서 미리 컴파일한
           커널 실행 파일(prebuilt.json)이 있으면 그것을, 없으면 cargo test를 사용
  docker : 시뮬레이터 컨테이너에 docker exec로 위임 (서버 운영 환경)

커널은 실행 시 작업 디렉토리 기준 data/*.bin 을 읽으므로, 동시에 두 커널이
같은 데이터 파일을 건드리지 않도록 모든 실행을 하나의 잠금으로 직렬화합니다.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path

from . import config

KERNEL_BACKEND = config.KERNEL_BACKEND
KERNEL_API = config.KERNEL_API

_RUN_LOCK = threading.Lock()
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")


def _check_name(name: str, what: str) -> str:
    if not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(f"허용되지 않는 {what} 이름: {name!r}")
    return name


# ── 입력 데이터 주입 ──────────────────────────────────────────────

@contextmanager
def _staged_inputs(inputs: dict[str, bytes] | None):
    """data/ 아래 파일을 임시로 교체하고, 끝나면 원래 상태로 되돌립니다."""
    if not inputs:
        yield []
        return

    data_dir = config.KERNEL_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    originals: dict[Path, bytes | None] = {}
    try:
        for name, payload in inputs.items():
            path = data_dir / _check_name(name, "데이터 파일")
            originals[path] = path.read_bytes() if path.exists() else None
            path.write_bytes(payload)
        yield list(inputs)
    finally:
        for path, original in originals.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)


# ── 실행 ──────────────────────────────────────────────────────────

def run_kernel(bin_name: str, timeout: int | None = None,
               inputs: dict[str, bytes] | None = None) -> tuple[bool, str, dict]:
    """커널 테스트 실행 → (passed, log, stats)"""
    _check_name(bin_name, "커널 bin")
    timeout = timeout or config.KERNEL_TIMEOUT

    if KERNEL_BACKEND not in ("local", "docker"):
        raise ValueError(f"알 수 없는 커널 백엔드: {KERNEL_BACKEND}")

    with _RUN_LOCK:
        try:
            with _staged_inputs(inputs) as staged:
                if KERNEL_BACKEND == "local":
                    r = _run_local(bin_name, timeout)
                else:
                    r = _run_docker(bin_name, timeout, staged)
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT", {}
        except Exception as e:  # noqa: BLE001 — 실행 실패는 결과로 보고
            return False, str(e), {}
        finally:
            if inputs and KERNEL_BACKEND == "docker":
                # 호스트에서 원본으로 복원된 파일을 컨테이너 작업 디렉토리에도 반영
                _docker_sync(list(inputs), timeout=30)

    return _parse_result(r)


def _cargo_test_cmd(bin_name: str) -> list[str]:
    return ["cargo", "test", "--release", "--bin", bin_name, "--", "--nocapture"]


def _prebuilt() -> dict:
    """빌드 단계에서 만든 bin 이름 → 테스트 실행 파일 목록 (없으면 빈 dict)"""
    path = config.KERNEL_DIR / "prebuilt.json"
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def _run_local(bin_name: str, timeout: int) -> subprocess.CompletedProcess:
    exe = _prebuilt().get(bin_name)
    if exe:
        # cargo test -- --nocapture 와 같은 테스트 실행 파일을 직접 실행 (Rust 툴체인 불필요)
        env = dict(os.environ)
        lib_dir = str(config.KERNEL_DIR / "lib")
        env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        return subprocess.run(
            [str(config.KERNEL_DIR / exe), "--nocapture"], cwd=config.KERNEL_DIR, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    return subprocess.run(
        _cargo_test_cmd(bin_name), cwd=config.KERNEL_DIR,
        capture_output=True, text=True, timeout=timeout,
    )


def _docker_sync(names: list[str], timeout: int) -> None:
    """호스트 data/ 의 지정 파일을 컨테이너 작업 디렉토리로 복사 (없어진 파일은 삭제)."""
    script = " ; ".join(
        f"if [ -f {config.DOCKER_SRC_DIR}/data/{n} ]; "
        f"then cp {config.DOCKER_SRC_DIR}/data/{n} {config.DOCKER_WORK_DIR}/data/{n}; "
        f"else rm -f {config.DOCKER_WORK_DIR}/data/{n}; fi"
        for n in map(shlex.quote, names)
    )
    subprocess.run(["docker", "exec", config.DOCKER_CONTAINER, "bash", "-c", script],
                   capture_output=True, timeout=timeout)


def _run_docker(bin_name: str, timeout: int, staged: list[str]) -> subprocess.CompletedProcess:
    if staged:
        _docker_sync(staged, timeout=30)
    cmd = f"cd {config.DOCKER_WORK_DIR} && {shlex.join(_cargo_test_cmd(bin_name))} 2>&1"
    return subprocess.run(
        ["docker", "exec", config.DOCKER_CONTAINER, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


# ── 결과 파싱 ─────────────────────────────────────────────────────

def _parse_result(r: subprocess.CompletedProcess) -> tuple[bool, str, dict]:
    log = (r.stdout or "") + (r.stderr or "")
    passed = "test result: ok" in log

    stats: dict = {}
    pairs = re.findall(r"PyTorch=\s*([-\d.e]+)\s+RNGD=\s*([-\d.e]+)\s+diff=([\d.e]+)", log)
    if pairs:
        pytorch_vals = [float(p[0]) for p in pairs]
        rngd_vals = [float(p[1]) for p in pairs]
        diffs = [float(p[2]) for p in pairs]
        stats["sample_pytorch"] = pytorch_vals[:4]
        stats["sample_rngd"] = rngd_vals[:4]
        stats["sample_diffs"] = diffs[:4]
        stats["max_sample_diff"] = max(diffs)
        stats["mean_diff"] = round(sum(diffs) / len(diffs), 6)

    return passed, log, stats


def backend_info() -> dict:
    """현재 커널 실행 환경 요약 (대시보드 표시용)."""
    info = {"backend": KERNEL_BACKEND, "kernel_api": KERNEL_API, "simulator": "furiosa-opt 0.6"}
    if KERNEL_BACKEND == "local":
        info["prebuilt_kernels"] = len(_prebuilt())
    if KERNEL_BACKEND == "docker":
        try:
            r = subprocess.run(["docker", "ps", "-f", f"name={config.DOCKER_CONTAINER}",
                                "--format", "{{.Status}}"], capture_output=True, text=True, timeout=5)
            info["container_status"] = r.stdout.strip() or "not running"
        except Exception:  # noqa: BLE001
            info["container_status"] = "unknown"
    return info


if __name__ == "__main__":
    print(f"Backend: {KERNEL_BACKEND} (kernel dir: {config.KERNEL_DIR})")
    ok, out, st = run_kernel("pilot_e2e_add")
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    if st:
        print(f"Stats: {st}")
