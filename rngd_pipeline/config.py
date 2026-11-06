"""
런타임 설정 — 경로, 커널 실행 백엔드, 기능 플래그를 한곳에서 관리합니다.

모든 값은 환경변수로 덮어쓸 수 있으며, 기본값은 Docker 체험 이미지 기준입니다.

    RNGD_KERNEL_BACKEND      local | docker            (기본: local)
    RNGD_KERNEL_DIR          Rust 커널 프로젝트 경로      (기본: <repo>/kernels)
    RNGD_KERNEL_TIMEOUT      커널 1회 실행 제한 시간(초)  (기본: 120)
    RNGD_DOCKER_CONTAINER    docker 백엔드 컨테이너 이름  (기본: furiosa-v06)
    RNGD_VAR_DIR             실행 이력·캐시 저장 경로     (기본: <repo>/var)
    RNGD_SCALES              사용할 검증 규모, 쉼표 구분  (기본: sim_small,sim_mid)
"""
import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 커널 실행 ─────────────────────────────────────────────────────
# local  : 같은 환경(체험 이미지 내부)에서 cargo를 직접 실행
# docker : 호스트에서 실행 중인 시뮬레이터 컨테이너에 docker exec로 위임
KERNEL_BACKEND = os.environ.get("RNGD_KERNEL_BACKEND", "local").strip().lower()

# 커널 소스가 따르는 furiosa-opt-std API 버전 (백엔드 종류와 무관)
KERNEL_API = "v06"

KERNEL_DIR = _env_path("RNGD_KERNEL_DIR", PROJECT_ROOT / "kernels")
KERNEL_DATA_DIR = KERNEL_DIR / "data"
KERNEL_TIMEOUT = int(os.environ.get("RNGD_KERNEL_TIMEOUT", "120"))

DOCKER_CONTAINER = os.environ.get("RNGD_DOCKER_CONTAINER", "furiosa-v06")
DOCKER_SRC_DIR = "/src"          # 컨테이너에 읽기 전용으로 마운트된 커널 소스
DOCKER_WORK_DIR = "/workspace"   # 컨테이너 내부 빌드 작업 디렉토리

# ── 실행 이력·캐시 ────────────────────────────────────────────────
VAR_DIR = _env_path("RNGD_VAR_DIR", PROJECT_ROOT / "var")
RESULTS_DB = VAR_DIR / "pipeline_results.db"
CACHE_DIR = VAR_DIR / "cache"

# ── 변환 규칙 레지스트리 ─────────────────────────────────────────
REGISTRY_SQL = PROJECT_ROOT / "registry" / "rngd_registry.sql"
REGISTRY_DB = VAR_DIR / "rngd_registry.db"

# ── 검증 규모 ─────────────────────────────────────────────────────
# sim_full(dim 4096)은 수백 MB의 커널 데이터가 필요해 기본 체험 범위에서 제외
AVAILABLE_SCALES = [s.strip() for s in os.environ.get("RNGD_SCALES", "sim_small,sim_mid").split(",") if s.strip()]


def ensure_dirs() -> None:
    """런타임에 쓰기가 필요한 디렉토리를 생성합니다."""
    for d in (VAR_DIR, CACHE_DIR, KERNEL_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
