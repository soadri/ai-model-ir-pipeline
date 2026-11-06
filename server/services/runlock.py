"""무거운 작업(통합 검증, 커스텀 모델 변환)을 한 번에 하나만 실행하기 위한 잠금."""
import threading

RUN_LOCK = threading.Lock()
BUSY = {"error": "다른 사용자가 작업을 실행 중입니다. 잠시 후 다시 시도해주세요.", "status": "busy"}
