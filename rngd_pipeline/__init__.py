"""
AI 모델 변환(표준 IR) 파이프라인

PyTorch 모델 → 표준 MLIR(linalg) → RNGD IR(rngd.* Dialect) 변환과
3레벨(구조 · 수치 · NPU 시뮬레이터) 검증을 제공합니다.
"""
__version__ = "0.1.0"
