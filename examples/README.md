# 사용자 모델 예시

대시보드의 사용자 모델 기능이나 명령줄로 바로 넣어볼 수 있는 작은 모델입니다.

| 예시 | 구조 | 식별되는 복합 패턴 |
|:--|:--|:--|
| [mlp.py](mlp.py) | Linear → SiLU → Linear | SiLU |
| [swiglu_ffn.py](swiglu_ffn.py) | gate·up 두 갈래 + SiLU + down (Llama 계열 FFN) | SiLU, FFN |

두 모델은 SiLU를 똑같이 쓰지만, 두 갈래를 곱하는 구조가 있는 SwiGLU만 FFN 패턴으로 식별됩니다.

## 실행

```bash
cd examples
docker run --rm -v "$PWD:/work" ghcr.io/soadri/ai-model-ir-pipeline:0.3 \
  python -m server.cli convert /work/mlp.py
```

대시보드에서는 「사용자 모델」 화면에 파일 내용을 붙여 넣으면 됩니다.

## 모델 파일 형식
- `model`: 변환할 `nn.Module` 객체
- `x`: 예시 입력 텐서
