#!/usr/bin/env python3
"""
cargo가 만든 커널 테스트 실행 파일을 모아 실행용 이미지로 옮길 준비를 합니다.

  1) `cargo test --no-run --message-format=json` 출력에서 bin별 테스트 실행 파일 경로를 찾아 복사
  2) 실행 파일이 동적으로 링크하는 라이브러리 중 시스템 기본 경로 밖(예: Rust 툴체인)에 있는 것만 복사
  3) bin 이름 → 실행 파일 경로를 prebuilt.json으로 기록 (kernel_backend가 읽음)

사용법: collect_kernels.py <cargo-json-출력> <출력-디렉토리>
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SYSTEM_LIB_DIRS = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")


def main() -> None:
    build_log, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "bin").mkdir(parents=True, exist_ok=True)
    (out / "lib").mkdir(parents=True, exist_ok=True)

    manifest = {}
    for line in build_log.read_text().splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (msg.get("reason") == "compiler-artifact" and msg.get("profile", {}).get("test")
                and msg.get("executable") and "bin" in msg.get("target", {}).get("kind", [])):
            name = msg["target"]["name"]
            dest = out / "bin" / name
            shutil.copy2(msg["executable"], dest)
            manifest[name] = f"bin/{name}"

    if not manifest:
        sys.exit("커널 테스트 실행 파일을 찾지 못했습니다. cargo 출력을 확인하세요.")

    copied = set()
    for rel in manifest.values():
        ldd = subprocess.run(["ldd", str(out / rel)], capture_output=True, text=True).stdout
        for so_name, so_path in re.findall(r"^\s*(\S+) => (/\S+)", ldd, re.M):
            if so_path.startswith(SYSTEM_LIB_DIRS) or so_name in copied:
                continue
            shutil.copy2(so_path, out / "lib" / so_name)
            copied.add(so_name)
        if "not found" in ldd:
            sys.exit(f"{rel}: 찾을 수 없는 공유 라이브러리가 있습니다\n{ldd}")

    (out / "prebuilt.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"커널 {len(manifest)}개: {', '.join(sorted(manifest))}")
    print(f"추가 공유 라이브러리 {len(copied)}개: {', '.join(sorted(copied)) or '없음'}")


if __name__ == "__main__":
    main()
