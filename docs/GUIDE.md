# 체험 가이드

AI 모델 변환(표준 IR) 파이프라인을 내 PC에서 직접 실행해보는 방법입니다. Docker만 있으면 되고, 처음 실행까지 10분 정도 걸립니다.

<br>

## 1. 준비

**Docker**가 필요합니다.

| 운영체제 | 설치할 것 |
|:--|:--|
| Windows 10/11 | [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) (WSL 2 기반으로 설치) |
| macOS | [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/) |
| Linux | [Docker Engine](https://docs.docker.com/engine/install/) |

설치 후 터미널(Windows는 PowerShell)에서 아래 명령이 버전을 출력하면 준비가 끝난 것입니다.

```bash
docker --version
```

> [!NOTE]
> - 메모리 8GB 이상, 디스크 여유 공간 3GB 이상을 권장합니다.
> - 이미지는 `linux/amd64`용입니다. Apple Silicon Mac이나 ARM 기반 Windows 노트북에서는 에뮬레이션으로 실행되어 검증 시간이 몇 배 길어질 수 있습니다.
> - Docker Desktop은 일정 규모 이상의 기관에서 업무로 쓸 때 유료 구독이 필요할 수 있습니다. 소속 기관의 라이선스 정책을 확인하시고, 필요하면 무료 대안인 [Rancher Desktop](https://rancherdesktop.io/)이나 [Podman Desktop](https://podman-desktop.io/)을 사용하셔도 됩니다.

<br>

## 2. 실행

아래 한 줄로 이미지를 받고 체험 서버를 실행합니다. 처음에는 이미지(약 380MB)를 내려받느라 1~3분 정도 걸립니다.

```bash
docker run --rm -p 127.0.0.1:8100:8100 ghcr.io/soadri/ai-model-ir-pipeline:0.1
```

`Uvicorn running on http://0.0.0.0:8100` 메시지가 나오면 브라우저에서 **http://localhost:8100** 에 접속합니다. 종료하려면 터미널에서 `Ctrl+C`를 누르세요.

> [!WARNING]
> 사용자 모델 분석은 입력한 PyTorch 코드를 체험 서버 안에서 그대로 실행합니다. 위 명령처럼 `127.0.0.1`로만 열어서 본인 PC에서만 접속하세요.

<br>

## 3. 해볼 것

### ① 기본 모델 검증

왼쪽 **기본 모델** 탭에서 검증 규모를 고르고 **변환과 검증 실행**을 누릅니다.

| 검증 규모 | 정상 결과 | 소요 시간 |
|:--|:--|:--|
| 기본 규모 (dim 16) | 커버리지 100%, NPU dialect 연산 45개, Level 1-2 12/12, Level 3 5/5 | 약 15초 |
| 중간 규모 GQA (dim 64) | 커버리지 100%, NPU dialect 연산 50개, Level 1-2 12/12, Level 3 5/5 | 약 30초 |

결과 화면의 **변환 단계**에서 PyTorch 모델이 표준 MLIR(linalg)을 거쳐 NPU 특화 IR로 낮아지는 과정을, **IR 보기**에서 변환된 연산 목록과 IR 전체를 확인할 수 있습니다.

### ② 사용자 모델 분석

왼쪽 **사용자 모델** 탭에서 예시를 불러와 실행해봅니다. 변환이 모두 되는 모델과 일부만 되는 모델을 하나씩 해보면 차이를 볼 수 있습니다.

| 예시 | 예상 커버리지 | 볼 것 |
|:--|:--|:--|
| SwiGLU FFN · Llama | 100% | 모든 연산이 NPU dialect 연산으로 변환 |
| GELU FFN · GPT-2 | 25% | **아직 변환하지 못한 연산**에 GELU가 복합 연산으로 표시되고, 구성 연산과 이유가 함께 나옴 |
| GQA Attention · Llama | 100% | Attention의 transpose와 행렬 곱이 하나의 `batch_gemm`으로 결합 |

예시 코드를 고치거나 직접 작성한 모델을 넣어봐도 됩니다. `nn.Module`을 상속한 클래스를 정의하고 마지막에 `model`과 `x` 변수를 선언하면 됩니다.

<br>

## 4. 결과 읽는 법

| 항목 | 의미 |
|:--|:--|
| 변환 커버리지 | 연산 중 NPU 특화 IR로 변환된 비율. 100% 미만이면 변환 규칙이 없어 CPU에서 처리되는 연산이 있다는 뜻입니다. |
| Level 1-2 | 변환된 IR의 구조(shape, dtype, 연결 관계 등)와 표준 IR 실행 결과가 PyTorch와 일치하는지 확인합니다. |
| Level 3 | 생성된 커널을 NPU 시뮬레이터(Functional)에서 실행한 결과를 PyTorch 기준값과 비교합니다. 상대 오차가 연산 계열별 허용 기준(elementwise 1%, contraction 5%) 이내면 통과입니다. |

Level 3는 연산 결과의 정합성을 확인하는 단계이며, 실제 NPU에서의 실행 성능을 나타내지 않습니다.

<br>

## 5. 명령줄로 실행 (선택)

웹 화면 없이 바로 결과만 볼 수도 있습니다.

```bash
docker run --rm ghcr.io/soadri/ai-model-ir-pipeline:0.1 python -m server.cli verify --scale sim_mid
```

직접 작성한 모델 파일을 분석하려면 파일을 컨테이너에 연결합니다.

```bash
# macOS, Linux
docker run --rm -v "$PWD/my_model.py:/tmp/model.py:ro" \
  ghcr.io/soadri/ai-model-ir-pipeline:0.1 python -m server.cli convert /tmp/model.py
```

```powershell
# Windows PowerShell (줄바꿈 기호가 백틱 `)
docker run --rm -v "${PWD}\my_model.py:/tmp/model.py:ro" `
  ghcr.io/soadri/ai-model-ir-pipeline:0.1 python -m server.cli convert /tmp/model.py
```

<br>

## 6. 문제 해결

| 증상 | 해결 |
|:--|:--|
| `port is already allocated` | 8100번 포트를 다른 프로그램이 쓰고 있습니다. `-p 127.0.0.1:8200:8100`으로 바꾸고 http://localhost:8200 에 접속하세요. |
| 이미지를 받을 때 `denied`, 시간 초과 | 사내망에서 `ghcr.io` 접속이 막혀 있을 수 있습니다. 네트워크 담당자에게 허용을 요청하세요. |
| Windows에서 Docker가 시작되지 않음 | Docker Desktop 설정에서 **Use the WSL 2 based engine**이 켜져 있는지, Windows 기능에서 WSL이 설치되어 있는지 확인하세요. |
| 검증이 매우 느림 | ARM 기반 PC(Apple Silicon, Snapdragon)에서는 에뮬레이션으로 실행되어 느릴 수 있습니다. |
| "다른 작업이 실행 중" 안내 | 검증은 한 번에 하나씩 실행됩니다. 앞의 실행이 끝난 뒤 다시 시도하세요. |

<br>

## 7. 의견 남기기

체험해보신 결과나 개선 의견은 [체험 의견 남기기](https://github.com/soadri/ai-model-ir-pipeline/issues/new?template=feedback.yml)로 알려주세요. 변환되지 않는 연산이나 오류를 발견하셨다면 [문제 신고](https://github.com/soadri/ai-model-ir-pipeline/issues/new?template=bug.yml)에 사용한 모델 코드와 함께 남겨주시면 큰 도움이 됩니다.
