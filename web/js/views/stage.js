// 시작 화면과 실행 중 화면 — README 배너와 같은 세 층으로 변환 과정을 보여줍니다.
import { clear, h } from "../dom.js";

function layers() {
  const code = (...parts) => h("code", {}, parts.map((p) => (Array.isArray(p) ? h("span", { class: "mk" }, p[0]) : p)));
  return h("div", { class: "strata" },
    h("div", { class: "stratum slate" }, h("div", { class: "name" }, "PyTorch", h("small", {}, "nn.Module")),
      code("scores = q @ k.", ["transpose"], "(-2, -1)")),
    h("div", { class: "stratum blue" }, h("div", { class: "name" }, "표준 MLIR", h("small", {}, "linalg dialect")),
      h("div", {}, code("linalg.", ["transpose"], " ins(%k) …"), h("br"), code("linalg.batch_matmul ins(%q, %kt)"))),
    h("div", { class: "stratum violet" }, h("div", { class: "name" }, "NPU 특화 IR", h("small", {}, "NPU dialect")),
      code("rngd.batch_gemm {", ["transpose_rhs = true"], "}")));
}

export function renderIntro(root) {
  clear(root).append(h("div", { class: "intro" },
    h("h1", {}, "PyTorch 모델이 NPU 특화 IR로 낮아지는 과정을 직접 확인해보세요"),
    h("p", {}, "왼쪽에서 기본 모델이나 직접 작성한 모델을 선택하고 실행하면, 표준 MLIR(linalg)을 거쳐 NPU 특화 IR로 변환한 뒤 "
      + "IR 정합성(Level 1-2)과 NPU 시뮬레이터 실행 결과(Level 3)까지 검증합니다."),
    layers(),
    h("p", { class: "small" }, "예: Attention의 transpose와 batch_matmul 연산 체인은 transpose_rhs 속성을 가진 하나의 NPU 연산으로 결합됩니다.")));
}

export function renderProgress(root, label, estimate) {
  const started = Date.now();
  const elapsed = h("span", { class: "elapsed" }, "0초");
  clear(root).append(h("div", { class: "intro progress", "aria-live": "polite" },
    h("h1", {}, label),
    h("p", {}, "표준 IR 변환, NPU 특화 IR 변환, Level 1-2 검증, NPU 시뮬레이터 실행을 차례로 진행합니다. ",
      estimate ? `보통 ${estimate} 걸립니다. ` : "", "경과 ", elapsed),
    layers()));
  const timer = setInterval(() => { elapsed.textContent = `${Math.round((Date.now() - started) / 1000)}초`; }, 1000);
  return () => clearInterval(timer);
}

export function renderError(root, message, detail) {
  clear(root).append(h("div", { class: "error", role: "alert" }, message, detail ? h("pre", {}, detail) : null));
}
