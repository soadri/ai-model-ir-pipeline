// 대시보드 진입점
import { api } from "./api.js";
import { h } from "./dom.js";
import { SCALES } from "./labels.js";
import { renderHistory } from "./views/history.js";
import { renderInput } from "./views/input.js";
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
  env.append(
    h("span", { class: "ok" }, `v${info.version || "0.1.0"}`),
    h("span", {}, info.kernel?.simulator || "NPU 시뮬레이터"),
    info.kernel?.prebuilt_kernels ? h("span", {}, `커널 ${info.kernel.prebuilt_kernels}개 준비됨`) : null);

  const input = renderInput(document.getElementById("input"), {
    scales: info.scales,
    onScale: (scale) => showCached(scale),
    onPreset: async (scale) => {
      const data = await run(() => api.verify(scale), "Llama-3.1 8B Decoder Layer를 변환하고 검증하는 중입니다", SCALES[scale]?.time, input);
      if (data) renderResult(main, normalizePreset(data, scale), await latestRunId());
    },
    onCustom: async (code) => {
      if (!code.trim()) return renderError(main, "모델 코드를 입력해주세요.");
      const data = await run(() => api.convert(code), "사용자 모델을 변환하고 검증하는 중입니다", "10초 안팎", input);
      if (data) renderResult(main, normalizeCustom(data), await latestRunId());
    },
  });

  renderHistory(historyList);

  if (api.demo === "custom") {
    input.select("custom");
    renderResult(main, normalizeCustom(await api.convert("")));
  } else {
    await showCached(input.selectedScale());
  }
}

init();
