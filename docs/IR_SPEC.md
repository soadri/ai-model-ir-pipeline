# NPU 특화 IR(RNGD Dialect) 명세

| 항목 | 내용 |
|:--|:--|
| 명세 버전 | 0.1 |
| 기준 구현 | 파이프라인 v0.1.2 (`rngd_pipeline/e2e_pipeline.py`의 `rewrite_to_rngd`) |
| 변환 규칙 | `registry/rngd_registry.sql` (25개 항목) |
| 작성 | 한국전자기술연구원(KETI) |

이 문서는 파이프라인이 표준 IR(MLIR linalg)에서 만들어 내는 NPU 특화 IR의 연산, 속성, 의미를 정의합니다. 문서의 모든 예시는 Llama-3.1 8B Decoder Layer 중간 규모 구성(dim 64, head 8개, KV head 2개)을 실제로 변환한 결과에서 가져왔습니다.

<br>

## 1. 개요

NPU 특화 IR은 표준 IR과 NPU 커널 사이에 두는 중간 표현입니다. 표준 IR의 연산 중 실제 계산을 하는 연산을 NPU가 처리하는 단위의 연산(`rngd.*`)으로 바꾸고, 그 밖의 연산(텐서 모양 변경, 상수, 모듈 구조)은 표준 IR 연산 그대로 둡니다. 따라서 NPU 특화 IR은 `rngd.*` 연산과 일부 표준 연산이 함께 있는 MLIR 모듈입니다(5절).

```
PyTorch 모델 ──torch-mlir──▶ 표준 IR (linalg, arith, math, tensor) ──재작성──▶ NPU 특화 IR (rngd.* + 남은 표준 연산) ──▶ vISA 커널
```

명세를 공개하는 목적은 두 가지입니다. 파이프라인 사용자가 변환 결과를 정확히 해석할 수 있게 하고, 다른 NPU를 대상으로 같은 구조의 Dialect와 백엔드를 만들려는 개발자에게 참조가 되게 하는 것입니다(6절).

<br>

## 2. 표기 규칙

- `rngd.*` 연산은 MLIR generic 형식으로 표기합니다: `%결과 = "rngd.연산"(%피연산자, ...) {속성} : (입력 타입) -> 출력 타입`
- 모든 속성 값은 문자열입니다. 목록은 `"[0, 2, 1, 3]"`, 참·거짓은 `"true"`처럼 씁니다.
- 텐서 타입은 표준 IR의 타입을 그대로 유지합니다(예: `tensor<1x8x64xf32>`). 변환 단계에서는 모델의 데이터 타입(f32)을 유지하고, 커널 실행 단계의 데이터 타입은 변환 규칙 레지스트리가 연산 계열별로 정합니다(7절).
- shape은 앞 축부터 0번으로 셉니다.

<br>

## 3. 연산 요약

| 연산 | 의미 | 피연산자 | 속성 | 원본 표준 연산 |
|:--|:--|:--|:--|:--|
| `rngd.gemm` | 행렬 곱 | 2개 | `transpose_rhs` (선택) | `linalg.matmul` |
| `rngd.batch_gemm` | 배치 행렬 곱 | 2개 | `transpose_rhs` (선택) | `linalg.batch_matmul` |
| `rngd.gemv` | 행렬·벡터 곱 | 2개 | 없음 | `linalg.matvec` |
| `rngd.dot_product` | 벡터 내적 | 2개 | 없음 | `linalg.dot` |
| `rngd.elementwise` | 원소 단위 연산 | 1~2개 | `op` | 원소 단위 `linalg.generic` |
| `rngd.reduce` | 축 방향 축소 | 1개 | `op` | 축약 축이 있는 `linalg.generic` |
| `rngd.transpose` | 축 순서 변경 | 1개 | `in_shape`, `out_shape`, `perm` | 입력을 그대로 내보내는 `linalg.generic` (축 순서가 바뀐 indexing map) |
| `rngd.broadcast` | 크기 1인 축 늘리기 | 1개 | `in_shape`, `out_shape`, `broadcast_dims` | 입력을 그대로 내보내는 `linalg.generic` (상수 축이 있는 indexing map) |

설계 단계에서 함께 정의했지만 현재 재작성에서 만들지 않는 연산(`rngd.fill`, `rngd.conv2d`)은 4.9절에 적었습니다.

<br>

## 4. 연산별 정의

### 4.1 `rngd.gemm`

2차원 행렬 곱 `C[M,N] = A[M,K] × B[K,N]`입니다.

| 구분 | 내용 |
|:--|:--|
| 피연산자 | `A` (M×K), `B` (K×N, `transpose_rhs`가 있으면 N×K) |
| 결과 | `C` (M×N) |
| 속성 | `transpose_rhs = "true"`: 두 번째 피연산자를 전치한 뒤 곱합니다. 속성이 없으면 전치하지 않습니다. |
| 변환 조건 | `linalg.matmul`. 출력 초기화(`linalg.fill`)는 흡수되어 사라집니다. |
| 전치 결합 | 두 번째 피연산자가 **마지막 두 축만 바꾸는 전치**(`linalg.generic`)에서 오면, 그 전치를 흡수하고 원래 텐서를 피연산자로 쓰며 `transpose_rhs`를 붙입니다. 전치와 행렬 곱 사이에 단순 복사만 있으면 함께 흡수하고, 확장이나 다른 축의 전치가 끼어 있으면 결합하지 않습니다. |

```mlir
// nn.Linear(64, 64): 가중치 전치, 출력 초기화, 행렬 곱의 세 연산이 하나로 결합됨
%29 = "rngd.gemm"(%28, %8) {transpose_rhs = "true"} : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
```

### 4.2 `rngd.batch_gemm`

배치 행렬 곱 `C[V,M,N] = A[V,M,K] × B[V,K,N]`입니다. 피연산자, 결과, `transpose_rhs`의 의미와 결합 조건은 `rngd.gemm`과 같고, 앞에 배치 축 V가 붙습니다.

```mlir
// Attention의 점수 계산 (Kᵀ는 앞의 rngd.transpose가 처리)
%69 = "rngd.batch_gemm"(%67, %68) : (tensor<8x8x8xf32>, tensor<8x8x8xf32>) -> tensor<8x8x8xf32>
```

### 4.3 `rngd.gemv`, 4.4 `rngd.dot_product`

각각 `linalg.matvec`(행렬·벡터 곱)과 `linalg.dot`(1차원 벡터 내적)에서 만들어집니다. 속성은 없습니다. Llama 계열 Decoder Layer에는 나타나지 않으며, 사용자 모델에서 `torch.mv`나 1차원 `torch.matmul`을 쓰는 경우에 나타납니다.

### 4.5 `rngd.elementwise`

원소마다 같은 계산을 하는 연산입니다. 어떤 계산인지는 `op` 속성으로 나타냅니다.

| `op` 값 | 계산 | 피연산자 | 원본 |
|:--|:--|:--|:--|
| `add`, `sub`, `mul`, `div` | 사칙 연산 | 2개 | `arith.addf`, `subf`, `mulf`, `divf` |
| `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sin`, `cos`, `erf` | 단항 수학 함수 | 1개 | `math.*` |
| `pow2` | 제곱 | 1개 | 지수가 2인 `math.powf` |
| `sigmoid` | 1 / (1 + e⁻ˣ) | 1개 | `arith.negf`, `math.exp`, `arith.addf`, `arith.divf`로 이루어진 본문 |

- **본문 판별**: `linalg.generic`의 본문(body)에 위 계산 중 하나만 있으면 해당 `op`로 바꿉니다. 본문이 여러 계산으로 이루어진 경우는 `sigmoid`만 인식하며, 그 밖의 조합은 변환하지 않고 미분류로 남깁니다(예: GELU).
- **크기가 다른 피연산자**: 두 피연산자의 shape이 다르면 크기 1인 축을 상대 쪽 크기로 늘려 계산합니다. 예를 들어 RMSNorm의 정규화는 `[1, 8, 64]`와 `[1, 8, 1]`을 곱합니다.
- **상수와의 연산**: torch-mlir가 상수와의 연산을 "상수로 채운 출력 텐서 + 입력 하나"로 표현하는 경우, 그 출력 텐서를 두 번째 피연산자로 바꿔 이항 연산으로 만듭니다.

```mlir
%19 = "rngd.elementwise"(%arg0) {op = "pow2"} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32>
%26 = "rngd.elementwise"(%arg0, %25) {op = "mul"} : (tensor<1x8x64xf32>, tensor<1x8x1xf32>) -> tensor<1x8x64xf32>
%103 = "rngd.elementwise"(%101) {op = "sigmoid"} : (tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
```

### 4.6 `rngd.reduce`

하나 이상의 축을 따라 값을 모으는 연산입니다.

| 구분 | 내용 |
|:--|:--|
| `op` | `sum` (본문이 `arith.addf`), `max` (본문이 `arith.maximumf` 또는 `arith.maxf`) |
| 결과 shape | 축소한 축의 크기가 1로 남습니다(축을 없애지 않음). |
| 축소 축 | 현재 속성으로 기록하지 않습니다. 입력과 출력 shape을 비교해 크기가 1로 줄어든 축으로 해석합니다(8절 한계 참고). |
| 실행 | 연동 중인 시뮬레이터 버전에서는 CPU 대체 처리합니다. |

```mlir
%22 = "rngd.reduce"(%19) {op = "sum"} : (tensor<1x8x64xf32>) -> tensor<1x8x1xf32>
%76 = "rngd.reduce"(%71) {op = "max"} : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x1xf32>
```

### 4.7 `rngd.transpose`

텐서의 축 순서를 바꿉니다. 값은 바꾸지 않습니다.

| 속성 | 의미 |
|:--|:--|
| `perm` | 출력의 j번째 축이 입력의 몇 번째 축인지 나타내는 목록. `perm = "[0, 2, 1, 3]"`이면 출력 축 순서는 입력의 (0, 2, 1, 3)번 축입니다. |
| `in_shape`, `out_shape` | 입력과 출력 shape. 확인용이며, 의미는 `perm`이 결정합니다. |

**판별 기준**: 입력을 그대로 내보내는 `linalg.generic`의 입력·출력 indexing map을 비교해, 모든 축이 일대일로 대응하면서 순서가 바뀌었으면 전치로 봅니다. shape만으로는 판별하지 않습니다. 축 크기가 같으면(예: `[1, 8, 8, 8]`) 전치와 다른 연산을 구분할 수 없기 때문입니다.

```mlir
// 어텐션 헤드 분할: [batch, seq, head, dim] → [batch, head, seq, dim]
%32 = "rngd.transpose"(%30) {in_shape = "[1, 8, 8, 8]", out_shape = "[1, 8, 8, 8]", perm = "[0, 2, 1, 3]"} : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
```

### 4.8 `rngd.broadcast`

크기가 1인 축을 늘려 같은 값을 반복합니다.

| 속성 | 의미 |
|:--|:--|
| `broadcast_dims` | 입력에서 크기 1이었다가 늘어나는 출력 축의 번호 목록 |
| `in_shape`, `out_shape` | 입력과 출력 shape |

**판별 기준**: 입력 indexing map에 상수(0) 축이 있거나 축이 모자라면 확장으로 봅니다. 입력과 출력의 indexing map이 같은 단순 복사도 `broadcast_dims = "[]"`인 `rngd.broadcast`로 표현합니다.

```mlir
// GQA: 키·값 헤드 2개를 질의 헤드 수에 맞춰 늘림
%60 = "rngd.broadcast"(%58) {broadcast_dims = "[2]", in_shape = "[1, 2, 1, 8, 8]", out_shape = "[1, 2, 4, 8, 8]"} : (tensor<1x2x1x8x8xf32>) -> tensor<1x2x4x8x8xf32>
// 크기 1인 배치 축 (실제 값은 바뀌지 않음)
%87 = "rngd.broadcast"(%86) {broadcast_dims = "[0]", in_shape = "[1, 8, 8, 8]", out_shape = "[1, 8, 8, 8]"} : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
```

### 4.9 설계에 정의되었으나 현재 만들지 않는 연산

| 연산 | 설계상 의미 | 현재 처리 |
|:--|:--|:--|
| `rngd.fill` | 상수 채우기·초기화 | 행렬 곱의 출력 초기화는 행렬 곱 연산에 흡수되고, 나머지 `linalg.fill`은 표준 연산으로 남습니다(5절). |
| `rngd.conv2d` | 2차원 합성곱 | 재작성 규칙이 없습니다. 합성곱 계열 모델 지원 시 추가합니다. |

<br>

## 5. NPU 특화 IR에 남는 표준 연산

재작성 후에도 다음 표준 연산은 그대로 남습니다. Llama-3.1 8B Decoder Layer 중간 규모 구성의 개수를 함께 적었습니다.

| 구분 | 연산 | 개수 | 의미 |
|:--|:--|:--:|:--|
| 데이터 배치 | `tensor.expand_shape`, `tensor.collapse_shape`, `tensor.extract_slice`, `tensor.concat`, `tensor.empty` | 38 | 텐서 모양만 바꾸거나 메모리를 준비하며, 계산은 없음 |
| 흡수 대상 | `arith.constant`, `linalg.fill` | 22 | 상수와 초기화. 인접 NPU 연산에 통합될 수 있음 |
| IR 구조 | `builtin.module`, `func.func`, `func.return`, `ml_program.global` | 4 | 모듈·함수 정의 |
| **변환 규칙 없음** | 부호를 뒤집는 `linalg.generic` (`arith.negf` 본문) | 2 | RoPE의 rotate_half. 8절 참고 |

<br>

## 6. 백엔드 인터페이스

재작성은 NPU 특화 IR과 함께, 바꾼 연산마다 커널 구성에 필요한 정보를 **커널 명세**(`RngdSpec`)로 남깁니다. NPU 백엔드는 IR을 다시 해석하지 않고 이 명세로 커널을 고르거나 만들 수 있습니다.

| 필드 | 의미 | 사용하는 연산 계열 |
|:--|:--|:--|
| `family` | 연산 계열: `elementwise`, `contraction`, `reduce`, `layout` | 전체 |
| `rngd_op` | 연산 이름 (예: `add`, `gemm`, `reduce_sum`, `transpose`) | 전체 |
| `shape` | 출력 shape | elementwise, reduce, layout |
| `batch`, `m`, `k`, `n` | 행렬 곱의 배치와 각 축 크기 | contraction |
| `reduce_axes` | 축소한 축 번호 | reduce |

**다른 NPU로 확장할 때**에는 표준 IR 변환, 연산 분류, 복합 패턴 식별, 검증 체계는 그대로 두고 다음 세 가지만 바꾸면 됩니다.

1. 대상 Dialect의 연산과 재작성 규칙 (`rewrite_to_rngd`에 해당하는 부분)
2. 변환 규칙 레지스트리의 대응 연산, 데이터 타입, 허용 오차
3. 커널 백엔드 (`rngd_pipeline/kernel_backend.py`): 커널 명세를 받아 해당 NPU의 커널을 실행하는 부분

<br>

## 7. 변환 규칙 레지스트리와 판정 기준

변환 규칙 레지스트리(`registry/rngd_registry.sql`)는 25개 항목으로, 각 항목에 원본 연산, 대응 NPU 특화 IR 연산, 실행 데이터 타입, 허용 오차와 그 근거를 담습니다. 연산 계열별 판정 기준은 다음과 같습니다.

| 연산 계열 | 항목 | 실행 데이터 타입 | 허용 오차 |
|:--|:--|:--|:--|
| contraction | gemm, batch_gemm, gemv, dot_product | bf16 | 절대 오차 0.5, 상대 오차 5% 이하 |
| elementwise | add, sub, mul, div, exp, log, sqrt, rsqrt, tanh, sin, cos, erf, pow2, sigmoid, silu, negf, gelu | f32 | 절대 오차 0.001, 상대 오차 1% 이하 (erf는 1e-4) |
| reduce | reduce_sum, reduce_max | f32 | 절대 오차 0.001, 상대 오차 1% 이하 |
| layout | transpose, broadcast | f32 | 값이 정확히 같아야 함 |

레지스트리의 `status`는 커널 쪽 지원 상태입니다. `gelu`는 `planned`(예정)이며, `silu`와 `negf`는 아래 8절처럼 재작성 규칙과 상태가 다를 수 있습니다.

<br>

## 8. 현재 한계

| 항목 | 내용 | 대응 계획 |
|:--|:--|:--|
| 부호 반전 미변환 | 레지스트리에는 `negf`가 지원으로 되어 있으나 재작성 규칙이 없어, RoPE의 부호 반전이 `linalg.generic`으로 남습니다. 연산 분류에서는 변환 대상으로 집계됩니다. | `rngd.elementwise`(`op = "negf"`) 재작성 규칙 추가 |
| 축소 축 미기록 | `rngd.reduce`는 축소 축을 속성으로 남기지 않습니다. 입력 축의 크기가 원래 1이면 IR만으로는 축소 축을 알 수 없습니다(커널 명세의 `reduce_axes`에는 남음). | `axes` 속성 추가 |
| 복합 본문 | 원소 단위 본문 중 `sigmoid` 외의 복합 계산(GELU 등)은 변환하지 않습니다. | 사용 빈도가 높은 조합부터 규칙 추가 |
| SiLU | SiLU는 `sigmoid`와 `mul` 두 연산으로 표현되며, 하나의 `op = "silu"` 연산으로 합치지는 않습니다. | 복합 패턴 식별 결과를 IR 표시로 연결할 때 함께 검토 |
| 미생성 연산 | `rngd.fill`, `rngd.conv2d`는 정의만 있고 만들지 않습니다(4.9절). | 대상 모델 확대 시 추가 |

<br>

## 9. 명세 버전 이력

| 명세 버전 | 파이프라인 | 변경 |
|:--|:--|:--|
| 0.1 | v0.1.2 | 첫 공개. `rngd.transpose`의 `perm`, `rngd.broadcast`의 `broadcast_dims` 포함 (v0.1.1까지는 두 속성이 없었고, 전치·확장을 shape으로 판별) |
