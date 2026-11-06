"""실행 이력 API — 목록, 상세, IR·커널 소스 다운로드."""
import re

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from server import storage

router = APIRouter(prefix="/api/verify/ir", tags=["runs"])
_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _download(text: str | None, filename: str, missing: str):
    if not text:
        return PlainTextResponse(missing, status_code=404)
    return PlainTextResponse(text, media_type="text/plain",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/runs")
def list_runs():
    """최근 실행 이력 20건"""
    return storage.list_runs()


@router.get("/runs/{run_id}")
def run_detail(run_id: str):
    """실행 상세 (변환 요약, 마커, 커널 목록)"""
    return storage.get_run(run_id) or {"error": "run_id not found"}


@router.get("/runs/{run_id}/rngd")
def download_rngd(run_id: str):
    """NPU 특화 IR 다운로드"""
    return _download(storage.get_run_text(run_id, "rngd"), f"rngd_ir_{run_id}.mlir", "IR not found")


@router.get("/runs/{run_id}/linalg")
def download_linalg(run_id: str):
    """표준 IR(linalg) 다운로드"""
    return _download(storage.get_run_text(run_id, "linalg"), f"linalg_ir_{run_id}.mlir", "IR not found")


@router.get("/runs/{run_id}/kernels/{module}")
def download_kernel(run_id: str, module: str):
    """커널 소스 다운로드"""
    if not (_SAFE.match(run_id) and _SAFE.match(module)):
        return PlainTextResponse("Kernel not found", status_code=404)
    return _download(storage.get_kernel_source(run_id, module), f"{module}_{run_id}.rs", "Kernel not found")
