# syntax=docker/dockerfile:1
#
# AI 모델 변환(표준 IR) 파이프라인 — 체험용 이미지
#
#   docker build -f docker/Dockerfile -t ai-model-ir-pipeline:dev .
#   docker run --rm -p 127.0.0.1:8100:8100 ai-model-ir-pipeline:dev
#
# torch·torch-mlir wheel은 원래 배포처에서 더 이상 받을 수 없는 버전이라, 이 저장소의
# GitHub Release(build-deps-1)에서 받아 SHA-256을 확인한 뒤 설치합니다 (docker/wheels.sha256 참고).
#
# 1단계에서 체험에 쓰는 커널만 미리 컴파일하고, 2단계(실행 이미지)에는 컴파일된 실행 파일만
# 옮겨 담기 때문에 실행 이미지에는 Rust 툴체인이 들어가지 않습니다.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ─────────────────────────────────────────────────────────────────
# 1단계: 커널 빌드
# ─────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS kernel-build

ARG RUST_TOOLCHAIN=nightly-2026-05-01
ARG CARGO_BUILD_JOBS=4
ARG TRIAL_KERNELS="pilot_golden_ffn_e2e pilot_golden_rmsnorm_e2e pilot_golden_attn_e2e pilot_golden_block_e2e \
pilot_golden_ffn_dim64 pilot_golden_rmsnorm_dim64 pilot_golden_attn_dim64 pilot_golden_block_dim64 \
pilot_golden_ffn_gate custom_verify_elementwise custom_verify_silu"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential pkg-config libssl-dev libclang-dev curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain none \
 && rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal -c rustfmt -c clippy

WORKDIR /build/kernels
COPY kernels/rust-toolchain.toml kernels/Cargo.toml kernels/Cargo.lock ./
COPY kernels/src ./src

# 체험용 커널의 테스트 실행 파일만 컴파일 (실행 파일 경로는 JSON으로 받아 다음 단계에서 수집)
RUN set -eu; bins=""; for b in ${TRIAL_KERNELS}; do bins="$bins --bin $b"; done; \
    cargo test --release --locked --no-run --message-format=json-render-diagnostics $bins \
      > /tmp/cargo-build.json

COPY docker/collect_kernels.py /tmp/collect_kernels.py
RUN python3 /tmp/collect_kernels.py /tmp/cargo-build.json /out

# 빌드 확인: 체험 데이터를 넣고 커널 하나를 실제로 실행
COPY kernels/data ./data
RUN out=$(LD_LIBRARY_PATH=/out/lib /out/bin/pilot_golden_ffn_e2e --nocapture 2>&1); \
    echo "$out" | tail -5; echo "$out" | grep -q "test result: ok"

# ─────────────────────────────────────────────────────────────────
# wheel 받기: Release에서 내려받으며 체크섬이 다르면 빌드가 실패합니다
# ─────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS wheels
ARG DEPS_URL=https://github.com/soadri/ai-model-ir-pipeline/releases/download/build-deps-1
ADD --checksum=sha256:083586533e4763ee4807cbb7bafb3beb52acc603c610f436781146d98626efed \
    ${DEPS_URL}/torch-2.3.0.dev20240122%2Bcpu-cp311-cp311-linux_x86_64.whl \
    /wheels/torch-2.3.0.dev20240122+cpu-cp311-cp311-linux_x86_64.whl
ADD --checksum=sha256:055d1c2684c10af1d77095ed6a973f2aaddf6edf32d29b9bb2b939f32455c971 \
    ${DEPS_URL}/torch_mlir-20240127.1096-cp311-cp311-linux_x86_64.whl \
    /wheels/torch_mlir-20240127.1096-cp311-cp311-linux_x86_64.whl

# ─────────────────────────────────────────────────────────────────
# 2단계: 실행 이미지
# ─────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="AI 모델 변환(표준 IR) 파이프라인" \
      org.opencontainers.image.description="PyTorch 모델을 표준 MLIR을 거쳐 NPU 특화 IR로 낮추고 NPU 시뮬레이터에서 검증" \
      org.opencontainers.image.source="https://github.com/soadri/ai-model-ir-pipeline" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Korea Electronics Technology Institute (KETI)"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/app

# Python 의존성 — wheel은 별도 단계에서 받은 것을 설치할 때만 연결해 최종 이미지에 남기지 않음
COPY requirements.txt constraints.txt ./
RUN --mount=type=bind,from=wheels,source=/wheels,target=/tmp/wheels \
    pip install /tmp/wheels/*.whl -r /opt/app/requirements.txt -c /opt/app/constraints.txt

# 미리 컴파일한 커널, 커널 소스(실행 이력의 소스 다운로드용), 체험 데이터, 파이프라인 코드
COPY --from=kernel-build /out/ ./kernels/
COPY kernels/Cargo.toml kernels/rust-toolchain.toml ./kernels/
COPY kernels/src/ ./kernels/src/
COPY kernels/data/ ./kernels/data/
COPY rngd_pipeline/ ./rngd_pipeline/
COPY server/ ./server/
COPY web/ ./web/
COPY registry/ ./registry/
COPY LICENSE NOTICE ./

# 일반 사용자 권한으로 실행. 커널 입력 데이터 주입과 실행 이력 저장 위치만 쓰기 허용
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data \
 && chown -R app:app /data /opt/app/kernels/data
USER app

ENV RNGD_HOST=0.0.0.0 RNGD_PORT=8100 \
    RNGD_KERNEL_BACKEND=local RNGD_KERNEL_DIR=/opt/app/kernels \
    RNGD_VAR_DIR=/data

EXPOSE 8100
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/api/info', timeout=4)" || exit 1

CMD ["python", "-m", "server"]
