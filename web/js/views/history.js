// 최근 실행 이력
import { api } from "../api.js";
import { clear, fmt, h } from "../dom.js";

export async function renderHistory(list) {
  let runs = [];
  try { runs = await api.runs(); } catch { /* 이력은 보조 정보라 실패해도 조용히 넘어감 */ }
  clear(list);
  if (!runs.length) {
    list.append(h("li", {}, h("span", { class: "small" }, "아직 실행 이력이 없습니다.")));
    return runs;
  }
  for (const r of runs.slice(0, 8)) {
    const l3 = r.total_l3 ? `L3 ${r.pass_l3}/${r.total_l3}` : "";
    list.append(h("li", {},
      h("span", {}, h("span", { class: "n" }, r.model_type === "custom" ? "사용자 모델" : "Llama-3.1 8B"),
        h("span", { class: "t" }, `  ${fmt.time(r.timestamp)}`)),
      h("span", { class: "r" }, `${fmt.pct(r.coverage_pct)}  ${l3}`),
      h("span", { class: "dl" },
        h("a", { href: api.url.rngd(r.run_id) }, "NPU 특화 IR"),
        h("a", { href: api.url.linalg(r.run_id) }, "표준 IR"))));
  }
  return runs;
}
