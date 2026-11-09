# 빌드 가이드

이 문서는 공개 소스에서 파이프라인을 직접 빌드하고 실행하는 방법을 설명합니다. 배포된 이미지로 체험만 하려면 [체험 가이드](GUIDE.md)를 보세요.

| 방법 | 이런 경우에 | 필요한 것 |
|:--|:--|:--|
| A. 체험 이미지를 직접 빌드 | 배포 이미지와 같은 환경을 소스에서 다시 만들 때 | Docker |
| B. 로컬 개발 환경 | 코드를 고치며 바로 실행해 볼 때, 기여할 때 | Linux x86_64, Python 3.11 (Level 3까지 하려면 Rust) |

<br>

## A. 체험 이미지를 직접 빌드

```bash
git clone https://github.com/soadri/ai-model-ir-pipeline.git
cd ai-model-ir-pipeline
docker build -f docker/Dockerfile -t ai-model-ir-pipeline:dev .
docker run --rm -p 127.0.0.1:8100:8100 ai-model-ir-pipeline:dev
```

빌드는 세 단계로 진행되며, 처음 빌드할 때는 커널 컴파일 때문에 PC 성능에 따라 수 분에서 20분 정도 걸립니다.

| 단계 | 하는 일 |
|:--|:--|
| 커널 빌드 | Rust 툴체인(nightly-2026-05-01)과 furiosa-opt로 체험용 커널 11개를 컴파일하고, 체험 데이터로 커널 하나를 실제로 실행해 확인 |
| wheel 받기 | 저장소의 [build-deps-1 릴리즈](https://github.com/soadri/ai-model-ir-pipeline/releases/tag/build-deps-1)에서 torch·torch-mlir wheel을 받음. SHA-256이 다르면 빌드가 멈춤 |
| 실행 이미지 | Python 의존성, 컴파일된 커널, 체험 데이터, 파이프라인 코드를 담음. Rust 툴체인은 포함하지 않음 |

빌드한 이미지로 검증까지 확인하려면 다음을 실행합니다. CI와 Release 워크플로도 같은 명령으로 검증합니다.

```bash
docker run --rm ai-model-ir-pipeline:dev python -m server.cli verify --scale sim_small
docker run --rm ai-model-ir-pipeline:dev python -m rngd_pipeline.layout_check
```

<br>

## B. 로컬 개발 환경

### 1. Python 환경

PyTorch와 torch-mlir는 검증에 사용한 버전으로 고정되어 있고, 그 wheel은 Linux x86_64용(Python 3.11)만 있습니다. Windows와 macOS에서는 A 방법이나 WSL2를 사용하세요.

```bash
git clone https://github.com/soadri/ai-model-ir-pipeline.git
cd ai-model-ir-pipeline
python3.11 -m venv .venv
source .venv/bin/activate

# 고정 wheel 받기와 체크섬 확인
mkdir -p wheels && cd wheels
DEPS=https://github.com/soadri/ai-model-ir-pipeline/releases/download/build-deps-1
curl -fL -o torch-2.3.0.dev20240122+cpu-cp311-cp311-linux_x86_64.whl "$DEPS/torch-2.3.0.dev20240122%2Bcpu-cp311-cp311-linux_x86_64.whl"
curl -fL -o torch_mlir-20240127.1096-cp311-cp311-linux_x86_64.whl "$DEPS/torch_mlir-20240127.1096-cp311-cp311-linux_x86_64.whl"
sha256sum -c ../docker/wheels.sha256
cd ..

pip install wheels/*.whl -r requirements.txt -c constraints.txt
```

`sha256sum -c`에서 두 파일 모두 `OK`가 나와야 합니다.

### 2. 변환과 Level 1-2 검증

여기까지만 준비해도 표준 IR 변환, NPU 특화 IR 재작성, Level 1-2 검증, 데이터 배치 연산 점검을 모두 실행할 수 있습니다.

```bash
python -m server.cli info                         # 버전과 실행 환경
python -m server.cli verify --scale sim_small     # 기본 모델 변환·검증
python -m server.cli convert my_model.py          # 사용자 모델 분석
python -m rngd_pipeline.layout_check              # 데이터 배치 연산 의미 점검
python -m server                                  # 웹 대시보드 (http://127.0.0.1:8100)
```

Rust 툴체인이 없으면 `verify`의 Level 3가 "실패"로 표시됩니다. 커널을 실행할 환경이 없어서 생기는 표시이며, Level 1-2 결과에는 영향이 없습니다.

### 3. Level 3 (NPU 시뮬레이터 검증)

Level 3는 Rust로 작성한 커널을 furiosa-opt 시뮬레이터로 실행합니다. 다음을 준비하면 파이프라인이 `kernels/` 디렉토리에서 `cargo test`로 커널을 실행합니다.

```bash
# 시스템 패키지 (Debian/Ubuntu)
sudo apt-get install -y build-essential pkg-config libssl-dev libclang-dev

# Rust: kernels/rust-toolchain.toml이 지정한 nightly-2026-05-01을 자동으로 사용
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"

python -m server.cli verify --scale sim_small
```

furiosa-opt(`furiosa-opt-std` 0.6)는 cargo가 crates.io에서 받으며, `kernels/Cargo.lock`으로 하위 의존성까지 고정되어 있습니다. 처음 실행할 때는 커널마다 컴파일이 일어나 시간이 걸리고, 이후에는 컴파일 결과를 재사용합니다.

### 4. 환경 변수

| 변수 | 의미 | 기본값 |
|:--|:--|:--|
| `RNGD_KERNEL_BACKEND` | 커널 실행 방식. `local`(같은 환경), `docker`(실행 중인 시뮬레이터 컨테이너에 위임) | `local` |
| `RNGD_KERNEL_DIR` | 커널 프로젝트 경로 | `<저장소>/kernels` |
| `RNGD_KERNEL_TIMEOUT` | 커널 1회 실행 제한 시간(초) | `120` |
| `RNGD_DOCKER_CONTAINER` | `docker` 방식에서 사용할 컨테이너 이름 | `furiosa-v06` |
| `RNGD_VAR_DIR` | 실행 이력과 캐시 저장 경로 | `<저장소>/var` |
| `RNGD_SCALES` | 사용할 검증 규모(쉼표로 구분) | `sim_small,sim_mid` |
| `RNGD_HOST`, `RNGD_PORT` | 대시보드 접속 주소 | `127.0.0.1`, `8100` |

<br>

## 문제 해결

| 증상 | 확인할 것 |
|:--|:--|
| `pip install`에서 torch를 찾지 못함 | Python 3.11과 Linux x86_64인지, wheel 두 개를 먼저 설치했는지 |
| `sha256sum -c`가 실패 | 내려받기가 끝까지 됐는지. 파일 크기는 torch 약 188MB, torch-mlir 약 50MB |
| Level 3가 모두 실패 | Rust 툴체인과 시스템 패키지가 있는지. `cargo --version`으로 확인 |
| 대시보드에 다른 PC에서 접속 안 됨 | 기본 접속 주소가 실행한 PC로 제한되어 있음. 사용자 모델 코드를 그대로 실행하므로 외부에 열 때는 주의 |

그 밖의 문제는 [문제 신고](https://github.com/soadri/ai-model-ir-pipeline/issues/new?template=bug.yml)로 알려 주세요.
