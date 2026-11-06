# 변경 이력

이 프로젝트의 버전별 변경 사항입니다. 버전 태그를 만들면 해당 버전의 내용이 Release 노트에 자동으로 들어갑니다.

## v0.1.1

- 대시보드 시작 화면의 변환 예시를 실제 파이프라인 출력 기준으로 바로잡았습니다. `nn.Linear`가 만드는 가중치 transpose와 `linalg.matmul`의 연산 체인이 `transpose_rhs` 속성을 가진 `rngd.gemm` 하나로 결합되는 예시입니다.
- 변경 이력(CHANGELOG)을 추가하고, 버전별 변경 사항이 Release 노트에 자동으로 포함되도록 했습니다.

## v0.1.0

첫 공개 릴리즈입니다.

- PyTorch 모델을 표준 MLIR(linalg)을 거쳐 NPU 특화 IR로 변환하고, Level 1-2(IR 정합성)와 Level 3(NPU 시뮬레이터 연산 정합성)으로 검증하는 파이프라인
- 웹 대시보드와 명령줄 도구가 포함된 체험용 Docker 이미지 (GHCR)
- 기본 모델 검증 규모: 기본(dim 16), 중간 GQA(dim 64)
- CI, Release, Upstream 변화 추적 워크플로
