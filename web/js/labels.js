// 화면에 표시하는 이름 — 내부 분류 키를 사용자가 이해할 수 있는 말로 바꿉니다.

export const MARKERS = [
  { key: "CONVERTED", label: "NPU 연산으로 변환", cls: "m-converted" },
  { key: "ABSORBABLE", label: "흡수 (상수, 초기화)", cls: "m-absorbable" },
  { key: "LAYOUT_TRANSFORM", label: "텐서 형태 변환", cls: "m-layout" },
  { key: "INFRASTRUCTURE", label: "모듈 구조", cls: "m-infra" },
  { key: "CPU_FALLBACK", label: "CPU 처리", cls: "m-fallback" },
  { key: "UNKNOWN", label: "미분류", cls: "m-unknown" },
];

export const PATTERNS = {
  RMSNORM: "RMSNorm", SOFTMAX: "Softmax", SILU: "SiLU", RESIDUAL: "Residual",
  ATTENTION: "Attention", FFN: "FFN", LAYERNORM: "LayerNorm", GELU: "GELU",
};

export const SCALES = {
  sim_small: { title: "기본 규모", detail: "dim 16, attention head 1", time: "약 15초" },
  sim_mid: { title: "중간 규모 (GQA)", detail: "dim 64, attention head 8, KV head 2", time: "약 30초" },
  sim_full: { title: "실규모", detail: "dim 4096, Llama-3.1 8B 1개 층", time: "수 분" },
};

export const STRATEGY = { pad: "패딩", tile: "K 분할(타일링)", direct: "그대로 실행" };
