"""
사용자 모델 변환 접근 토큰 (선택 기능)

공개판 체험 이미지처럼 본인 PC에서만 쓰는 경우에는 설정하지 않습니다(기본: 토큰 없이 실행).
여러 사람이 접속하는 서버에서 운영할 때, 아래 중 하나를 설정하면 토큰이 있어야 변환을 실행합니다.

    RNGD_CONVERT_TOKEN_FILE=/data/convert_tokens   # 파일: 한 줄에 「기관:토큰」, 요청마다 다시 읽음
    RNGD_CONVERT_TOKENS="nota:토큰1,postech:토큰2"  # 환경 변수 (바꾸면 서버 재시작 필요)

토큰 파일을 지정했는데 파일이 없거나 비어 있으면 변환을 막습니다(잠금이 기본).
요청마다 시각, 기관, 성공 여부, 접속 IP를 VAR_DIR/convert_access.log에 남깁니다.
"""
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from rngd_pipeline import config

HEADER = "x-access-token"
MESSAGE = "사용자 모델 변환은 접근 토큰이 필요합니다. 토큰은 서버 관리자에게 문의해 주세요."


def _token_file():
    path = os.environ.get("RNGD_CONVERT_TOKEN_FILE", "").strip()
    return Path(path) if path else None


def required() -> bool:
    """토큰 확인을 켰는지 (토큰 파일 경로나 토큰 목록 중 하나라도 설정)"""
    return _token_file() is not None or bool(os.environ.get("RNGD_CONVERT_TOKENS", "").strip())


def _parse(lines) -> dict:
    tokens = {}
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#") or ":" not in ln:
            continue
        org, tok = ln.split(":", 1)
        if tok.strip():
            tokens[tok.strip()] = org.strip()
    return tokens


def _tokens() -> dict:
    tokens = _parse(os.environ.get("RNGD_CONVERT_TOKENS", "").split(","))
    path = _token_file()
    if path is not None:
        try:
            tokens.update(_parse(path.read_text(encoding="utf-8").splitlines()))
        except FileNotFoundError:
            pass
    return tokens


def check(given: str):
    """맞는 토큰이면 기관 이름, 아니면 None"""
    given = (given or "").strip()
    if not given:
        return None
    for tok, org in _tokens().items():
        if hmac.compare_digest(given.encode("utf-8"), tok.encode("utf-8")):
            return org
    return None


def log(org, ok: bool, ip: str) -> None:
    try:
        config.VAR_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.VAR_DIR / "convert_access.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "org": org,
                                "ok": ok, "ip": ip}, ensure_ascii=False) + "\n")
    except OSError:
        pass
