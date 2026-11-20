"""
체험용 웹 서버 — 검증 API와 대시보드 화면을 제공합니다.

    python -m server            # 기본: 127.0.0.1:8100
    RNGD_HOST=0.0.0.0 python -m server   # Docker 컨테이너 안에서 실행할 때

사용자 모델 변환은 입력 코드를 서버 프로세스에서 실행하므로, 본인 PC에서만 사용하세요.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from rngd_pipeline import __version__, config
from rngd_pipeline.kernel_backend import backend_info
from server import storage
from server.routes import registry, runs, verify
from server.services import access

WEB_DIR = config.PROJECT_ROOT / "web"

app = FastAPI(title="AI 모델 변환(표준 IR) 파이프라인", version=__version__,
              docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
app.include_router(verify.router)
app.include_router(runs.router)
app.include_router(registry.router)


# 화면 파일은 브라우저가 쓸 때마다 바뀌었는지 확인하게 한다(Cache-Control: no-cache).
# 바뀌지 않았으면 ETag로 304만 받아 다시 받지 않으므로 속도 영향은 거의 없고,
# 서버를 새 버전으로 올렸을 때 예전 화면 파일이 남아 새 결과 항목이 보이지 않는 일을 막는다.
_NO_CACHE_PREFIXES = ("/static/", "/ext/")


@app.middleware("http")
async def _revalidate_web_files(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith(_NO_CACHE_PREFIXES):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.on_event("startup")
def _startup():
    config.ensure_dirs()
    storage.init_db()


@app.get("/api/info", tags=["info"])
def info():
    """버전, 커널 실행 환경, 사용 가능한 검증 규모, 사용자 모델 변환의 토큰 필요 여부"""
    return {"version": __version__, "kernel": backend_info(), "scales": config.AVAILABLE_SCALES,
            "convert_token_required": access.required()}


if (WEB_DIR / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
