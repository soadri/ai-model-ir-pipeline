// 대시보드 진입점
import { api } from "./api.js";
import { h } from "./dom.js";
import { SCALES } from "./labels.js";
import { renderHistory } from "./views/history.js";
import { renderInput } from "./views/input.js";
import { renderRegistry } from "./views/registry.js";
import { normalizeCustom, normalizePreset, renderResult } from "./views/result.js";
import { renderError, renderIntro, renderProgress } from "./views/stage.js";

const main = document.getElementById("main");
const historyList = document.getElementById("history");
let busy = false;

async function latestRunId() {
  const runs = await renderHistory(historyList);
  return runs[0]?.run_id;
}

function failed(data) {
  if (data?.status === "busy") return "다른 작업이 실행 중입니다. 잠시 후 다시 실행해주세요.";
  if (data?.error) return data.error;
  return null;
}

async function showCached(scale) {
  if (busy) return;
  try {
    const cached = await api.cached(scale);
    if (cached?._from_cache && cached.level12) renderResult(main, normalizePreset(cached, scale));
    else renderIntro(main);
  } catch { renderIntro(main); }
}

async function run(task, label, estimate, input) {
  if (busy) return;
  busy = true;
  input.setBusy(true);
  const stop = renderProgress(main, label, estimate);
  try {
    const data = await task();
    stop();
    const message = failed(data);
    if (message) return renderError(main, "실행하지 못했습니다.", message);
    return data;
  } catch (e) {
    stop();
    renderError(main, "서버와 통신하지 못했습니다. 체험 서버가 실행 중인지 확인해주세요.", String(e));
  } finally {
    busy = false;
    input.setBusy(false);
  }
}

async function init() {
  const info = await api.info().catch(() => ({ scales: ["sim_small"], kernel: {} }));
  const env = document.getElementById("env");
  env.append(...[
    h("span", { class: "ok" }, `v${info.version || "0.1.0"}`),
    h("span", {}, info.kernel?.simulator || "NPU 시뮬레이터"),
    info.kernel?.prebuilt_kernels ? h("span", {}, `커널 ${info.kernel.prebuilt_kernels}개 준비됨`) : null,
  ].filter(Boolean));

  const input = renderInput(document.getElementById("input"), {
    scales: info.scales,
    tokenRequired: !!info.convert_token_required,
    onScale: (scale) => showCached(scale),
    onPreset: async (scale) => {
      const data = await run(() => api.verify(scale), "Llama-3.1 8B Decoder Layer를 변환하고 검증하는 중입니다", SCALES[scale]?.time, input);
      if (data) renderResult(main, normalizePreset(data, scale), await latestRunId());
    },
    onCustom: async (code, token) => {
      if (!code.trim()) return renderError(main, "모델 코드를 입력해주세요.");
      if (info.convert_token_required && !token) return renderError(main, "접근 토큰을 입력해주세요.", "이 서버는 토큰이 있어야 사용자 모델을 실행합니다. 토큰은 서버 관리자에게 문의해 주세요.");
      const data = await run(() => api.convert(code, token), "사용자 모델을 변환하고 검증하는 중입니다", "10초 안팎", input);
      if (data) renderResult(main, normalizeCustom(data), await latestRunId());
    },
  });

  renderHistory(historyList);

  // 상단 화면 전환: #rules 이면 변환 규칙, 그 밖은 변환·검증
  const nav = document.getElementById("nav");
  const views = { verify: "변환·검증", rules: "변환 규칙" };
  const navButtons = Object.entries(views).map(([key, label]) =>
    h("a", { href: key === "verify" ? "#" : `#${key}`, dataset: { view: key } }, label));
  nav.append(...navButtons);
  let lastVerify = null;
  const show = async () => {
    const view = location.hash === "#rules" ? "rules" : "verify";
    document.body.dataset.view = view;
    for (const b of navButtons) b.setAttribute("aria-current", b.dataset.view === view ? "page" : "false");
    if (view === "rules") {
      if (!busy) { lastVerify = [...main.childNodes]; await renderRegistry(main); }
    } else if (lastVerify) {
      main.replaceChildren(...lastVerify);
      lastVerify = null;
    }
  };
  window.addEventListener("hashchange", show);
  if (location.hash === "#rules") return show();
  document.body.dataset.view = "verify";
  navButtons[0].setAttribute("aria-current", "page");

  if (api.demo === "custom") {
    input.select("custom");
    renderResult(main, normalizeCustom(await api.convert("")));
  } else {
    await showCached(input.selectedScale());
  }
}

init();
