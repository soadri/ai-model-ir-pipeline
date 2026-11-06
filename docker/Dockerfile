# syntax=docker/dockerfile:1
#
# AI 모델 변환(표준 IR) 파이프라인 — 체험용 이미지
#
#   docker build -f docker/Dockerfile -t ai-model-ir-pipeline:dev .
#   docker run --rm -p 127.0.0.1:8100:8100 ai-model-ir-pipeline:dev
#
# 빌드 전에 docker/wheels/ 에 torch·torch-mlir wheel 두 개를 넣어야 합니다.
# (원래 배포처에서 더 이상 받을 수 없는 버전이라 별도로 제공하며, 체크섬은 docker/wheels.sha256)
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

# Python 의존성 — 재구성 wheel은 체크섬을 확인한 뒤 설치
COPY requirements.txt constraints.txt ./
COPY docker/wheels/ /tmp/wheels/
COPY docker/wheels.sha256 /tmp/wheels/wheels.sha256
RUN cd /tmp/wheels && sha256sum -c wheels.sha256 \
 && pip install /tmp/wheels/*.whl -r /opt/app/requirements.txt -c /opt/app/constraints.txt \
 && rm -rf /tmp/wheels

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
