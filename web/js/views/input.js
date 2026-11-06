// 입력 패널 — 기본 모델 / 사용자 모델
import { h } from "../dom.js";
import { EXAMPLES } from "../examples.js";
import { SCALES } from "../labels.js";

export function renderInput(root, { scales, onPreset, onCustom, onScale }) {
  const presetBtn = h("button", { role: "tab", "aria-selected": "true" }, "기본 모델");
  const customBtn = h("button", { role: "tab", "aria-selected": "false" }, "사용자 모델");

  // 기본 모델
  const scaleList = h("div", { class: "scales", role: "radiogroup", "aria-label": "검증 규모" },
    scales.map((s, i) => h("label", { class: "scale" },
      h("input", { type: "radio", name: "scale", value: s, checked: i === 0, onchange: () => onScale(s) }),
      h("b", {}, SCALES[s]?.title || s),
      h("span", { class: "small" }, `${SCALES[s]?.detail || ""}, ${SCALES[s]?.time || ""}`))));
  const presetRun = h("button", { class: "run", onclick: () => onPreset(selectedScale()) }, "변환과 검증 실행");
  const presetPane = h("div", {},
    h("div", { class: "model-card" },
      h("strong", {}, "Llama-3.1 8B Decoder Layer"),
      h("div", { class: "small" }, "RMSNorm, GQA Attention(RoPE), SwiGLU FFN, Residual로 구성된 한 개 층")),
    h("span", { class: "field-label" }, "검증 규모"),
    scaleList, presetRun);
  const selectedScale = () => scaleList.querySelector("input:checked")?.value || scales[0];

  // 사용자 모델
  const editor = h("textarea", { class: "editor", spellcheck: "false", "aria-label": "PyTorch 모델 코드" });
  const chips = EXAMPLES.map((ex) => h("button", {
    class: "chip", "aria-pressed": "false", title: ex.expected ? `기존 검증 기준 변환 커버리지 ${ex.expected}` : null,
    onclick: (e) => {
      editor.value = ex.code;
      for (const c of chips) c.setAttribute("aria-pressed", String(c === e.currentTarget));
    },
  }, ex.title, ex.expected ? h("em", { class: ex.expected === "100%" ? "full" : "" }, ex.expected) : null));
  const customRun = h("button", { class: "run", onclick: () => onCustom(editor.value) }, "변환 분석 실행");
  const customPane = h("div", { hidden: true },
    h("span", { class: "field-label", style: "margin-top:0" }, "예시 불러오기"),
    h("div", { class: "examples" }, chips),
    editor,
    h("ol", { class: "rules" },
      h("li", {}, h("code", {}, "nn.Module"), "을 상속한 모델 클래스를 정의합니다."),
      h("li", {}, "마지막에 ", h("code", {}, "model = MyModel()"), "과 ", h("code", {}, "x = torch.randn(...)"), "을 선언합니다."),
      h("li", {}, "가중치는 무작위 초기화로 충분합니다. 변환 결과는 연산 구조로 결정됩니다.")),
    customRun,
    h("p", { class: "small", style: "margin:10px 0 0" }, "입력한 코드는 이 체험 서버(내 PC의 컨테이너) 안에서 실행됩니다."));
  editor.value = EXAMPLES[0].code;
  chips[0].setAttribute("aria-pressed", "true");

  const select = (tab) => {
    const isPreset = tab === "preset";
    presetBtn.setAttribute("aria-selected", String(isPreset));
    customBtn.setAttribute("aria-selected", String(!isPreset));
    presetPane.hidden = !isPreset;
    customPane.hidden = isPreset;
  };
  presetBtn.onclick = () => select("preset");
  customBtn.onclick = () => select("custom");

  root.append(h("section", { class: "panel" },
    h("h2", {}, "모델 선택"),
    h("p", { class: "hint" }, "검증된 Llama 구조로 바로 실행하거나, 직접 작성한 PyTorch 모델을 분석할 수 있습니다."),
    h("div", { class: "tabs", role: "tablist" }, presetBtn, customBtn),
    presetPane, customPane));

  return {
    select,
    selectedScale,
    setBusy(busy) {
      presetRun.disabled = customRun.disabled = busy;
      presetRun.textContent = busy ? "실행 중…" : "변환과 검증 실행";
      customRun.textContent = busy ? "실행 중…" : "변환 분석 실행";
    },
  };
}
