# 변경 이력

이 프로젝트의 버전별 변경 사항입니다. 버전 태그를 만들면 해당 버전의 내용이 Release 노트에 자동으로 들어갑니다.

## 다음 버전 (v0.2.0 예정)

- NPU 특화 IR 연산에 최적화 마커를 기록합니다. 복합 패턴 식별 결과(`rngd.fusion`), 그 범위를 대체할 수 있는 특화 커널 후보(`rngd.kernel_hint`), 현재 CPU에서 대체 처리하는 연산(`rngd.exec = "cpu_fallback"`)을 연산 속성으로 남겨, IR만 보고 특화 커널로 바꿀 범위와 NPU에서 직접 실행하지 않는 연산을 알 수 있게 했습니다. 보조 속성이라 검증 결과, 연산 분류, 복합 패턴 식별 결과는 이전과 같습니다.
- NPU 특화 IR 명세를 0.2로 갱신했습니다(최적화 마커 절 추가).

## v0.1.2

- NPU 특화 IR 재작성에서 데이터 배치 연산(전치, 확장)을 판별하는 기준을 입력·출력 shape에서 indexing_maps로 바꿨습니다. 이전에는 축 크기가 같으면 전치를 구분하지 못해, 중간 규모 기본 모델에서 Attention의 Kᵀ 등 전치 3개가 `rngd.broadcast`로, GQA의 키·값 헤드 확장 2개가 `rngd.transpose`로 표시되었습니다. 검증 결과(Level 1-2, Level 3)와 연산 분류는 이 수정 전후로 같습니다.
- `rngd.transpose`에 축 순서(`perm`), `rngd.broadcast`에 늘어난 축(`broadcast_dims`)을 기록해, NPU 특화 IR만으로 데이터 배치를 정확히 알 수 있게 했습니다.
- 행렬 곱의 전치 체인 결합은 마지막 두 축만 바꾸는 전치일 때만 적용하고, 중간에 확장이 끼면 결합하지 않도록 했습니다.
- 데이터 배치 연산의 의미를 확인하는 보조 검사(`python -m rngd_pipeline.layout_check`)를 추가하고, CI와 Release 검증에 포함했습니다.

## v0.1.1

- 대시보드 시작 화면의 변환 예시를 실제 파이프라인 출력 기준으로 바로잡았습니다. `nn.Linear`가 만드는 가중치 transpose와 `linalg.matmul`의 연산 체인이 `transpose_rhs` 속성을 가진 `rngd.gemm` 하나로 결합되는 예시입니다.
- 변경 이력(CHANGELOG)을 추가하고, 버전별 변경 사항이 Release 노트에 자동으로 포함되도록 했습니다.

## v0.1.0

첫 공개 릴리즈입니다.

- PyTorch 모델을 표준 MLIR(linalg)을 거쳐 NPU 특화 IR로 변환하고, Level 1-2(IR 정합성)와 Level 3(NPU 시뮬레이터 연산 정합성)으로 검증하는 파이프라인
- 웹 대시보드와 명령줄 도구가 포함된 체험용 Docker 이미지 (GHCR)
- 기본 모델 검증 규모: 기본(dim 16), 중간 GQA(dim 64)
- CI, Release, Upstream 변화 추적 워크플로
