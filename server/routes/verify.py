"""검증 API — 기본 모델 통합 검증, 캐시 조회, 사용자 모델 변환."""
import time

from fastapi import APIRouter, Body
from fastapi.concurrency import run_in_threadpool

from rngd_pipeline import config
from server.services import custom_convert, preset_verify
from server.services.runlock import BUSY, RUN_LOCK

router = APIRouter(prefix="/api/verify", tags=["verify"])


def _all_verify(scale: str) -> dict:
    if not RUN_LOCK.acquire(blocking=False):
        return BUSY
    try:
        return preset_verify.run_all(scale=scale)
    finally:
        RUN_LOCK.release()


@router.get("/all_verify")
async def all_verify(scale: str = preset_verify.DEFAULT_SCALE):
    """기본 모델 Level 1-2-3 통합 검증"""
    if scale not in preset_verify.SCALES or scale not in config.AVAILABLE_SCALES:
        return {"error": f"이 환경에서 사용할 수 없는 검증 규모입니다: {scale}",
                "available": config.AVAILABLE_SCALES}
    return await run_in_threadpool(_all_verify, scale)


@router.get("/cached_result")
def cached_result(scale: str = preset_verify.DEFAULT_SCALE):
    """마지막 통합 검증 결과 (없으면 _from_cache=false)"""
    data = preset_verify.load_cache(scale) if scale in preset_verify.SCALES else None
    if data:
        data["_from_cache"] = True
        return data
    return {"_from_cache": False, "scale": scale}


def _custom_convert(code: str) -> dict:
    started = time.time()
    try:
        model, x = custom_convert.prepare(code)
    except custom_convert.InputError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001 — 사용자 코드 실행 오류
        return {"error": f"{type(e).__name__}: {e}"}

    if not RUN_LOCK.acquire(blocking=False):
        return BUSY
    try:
        return custom_convert.analyze(model, x, code, started)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        RUN_LOCK.release()


@router.post("/custom_convert")
async def custom_convert_api(payload: dict = Body(...)):
    """사용자 정의 PyTorch 모델 변환·검증"""
    return await run_in_threadpool(_custom_convert, str(payload.get("code", "")))
