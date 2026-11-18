// 변환 규칙 레지스트리 — 표준 IR 연산이 어떤 NPU 특화 IR 연산으로 바뀌고, 무엇으로 검증하는지
import { api } from "../api.js";
import { clear, h, icon } from "../dom.js";
import { FAMILIES, RULE_STATUS } from "../labels.js";

const SEARCH_FIELDS = ["op_name", "source_op", "target_op", "detail", "compares"];

export async function renderRegistry(main) {
  clear(main);
  let rules = [];
  try {
    rules = (await api.registry()).rules || [];
  } catch (e) {
    main.append(h("section", { class: "panel error" }, h("h2", {}, "변환 규칙을 불러오지 못했습니다."), h("p", { class: "hint" }, String(e))));
    return;
  }

  const state = { q: "", family: "", dtype: "", open: null };
  const count = h("span", { class: "small" });
  const body = h("tbody");

  const search = h("input", {
    class: "search", type: "search", placeholder: "연산 이름, IR 연산, 커널 체인으로 검색 (예: rsqrt, batch_matmul, Mode10)",
    "aria-label": "변환 규칙 검색",
    oninput: (e) => { state.q = e.target.value.trim().toLowerCase(); draw(); },
  });

  const chipGroup = (key, options) => {
    const chips = options.map(([value, label]) => h("button", {
      class: "chip", "aria-pressed": String(state[key] === value),
      onclick: () => {
        state[key] = value;
        for (const c of chips) c.setAttribute("aria-pressed", String(c === chipsByValue[value]));
        draw();
      },
    }, label));
    const chipsByValue = Object.fromEntries(options.map(([v], i) => [v, chips[i]]));
    return h("div", { class: "examples", role: "group" }, chips);
  };
  const dtypes = [...new Set(rules.map((r) => r.dtype))].sort();

  function matches(r) {
    if (state.family && r.family !== state.family) return false;
    if (state.dtype && r.dtype !== state.dtype) return false;
    if (state.q && !SEARCH_FIELDS.some((f) => (r[f] || "").toLowerCase().includes(state.q))) return false;
    return true;
  }

  function detailRow(r) {
    const item = (label, value, cls) => value ? h("div", { class: `rule-item ${cls || ""}` }, h("span", { class: "k" }, label), h("span", { class: "v" }, value)) : null;
    return h("tr", { class: "rule-detail" }, h("td", { colspan: "6" },
      h("div", { class: "rule-grid" },
        item("비교 대상", r.compares),
        item("커널 구현", r.detail, "mono"),
        item("통과의 의미", r.meaning_pass, "pass"),
        item("실패의 의미", r.meaning_fail, "fail"))));
  }

  function draw() {
    clear(body);
    const shown = rules.filter(matches);
    count.textContent = `${shown.length}개 / 전체 ${rules.length}개`;
    if (!shown.length) {
      body.append(h("tr", {}, h("td", { colspan: "6", class: "small" }, "조건에 맞는 변환 규칙이 없습니다.")));
      return;
    }
    for (const r of shown) {
      const st = RULE_STATUS[r.status] || { label: r.status, cls: "" };
      const open = state.open === r.op_name;
      body.append(h("tr", {
        class: `rule-row${open ? " open" : ""}`, tabindex: "0", "aria-expanded": String(open),
        onclick: () => { state.open = open ? null : r.op_name; draw(); },
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); state.open = open ? null : r.op_name; draw(); } },
      },
        h("td", {}, h("b", {}, r.op_name), r.experimental ? h("span", { class: "tag exp" }, "실험") : null),
        h("td", {}, FAMILIES[r.family]?.label || r.family),
        h("td", {}, h("code", {}, r.source_op)),
        h("td", {}, h("code", { class: "npu" }, r.target_op)),
        h("td", {}, r.dtype, h("span", { class: "small" }, r.tolerance)),
        h("td", {}, h("span", { class: `state ${st.cls}` }, st.label))));
      if (open) body.append(detailRow(r));
    }
  }

  main.append(
    h("div", { class: "result-head" },
      h("h1", {}, "변환 규칙"),
      h("div", { class: "meta" }, "표준 IR(linalg) 연산을 NPU 특화 IR 연산으로 바꾸는 규칙과, 각 규칙을 NPU 시뮬레이터에서 무엇과 비교해 검증하는지 정리한 레지스트리입니다.")),
    h("section", { class: "panel" },
      h("div", { class: "rules-toolbar" },
        h("div", { class: "search-wrap" }, icon("search"), search),
        h("div", { class: "filters" },
          h("span", { class: "field-label" }, "계열"),
          chipGroup("family", [["", "전체"], ...Object.entries(FAMILIES).map(([k, v]) => [k, v.label])]),
          h("span", { class: "field-label" }, "정밀도"),
          chipGroup("dtype", [["", "전체"], ...dtypes.map((d) => [d, d])]),
          h("span", { class: "spacer" }), count)),
      h("div", { class: "table-wrap" },
        h("table", { class: "t rules-table" },
          h("thead", {}, h("tr", {},
            h("th", {}, "연산"), h("th", {}, "계열"), h("th", {}, "표준 IR 연산"), h("th", {}, "NPU 특화 IR 연산"),
            h("th", {}, "정밀도 · 허용 오차"), h("th", {}, "상태"))),
          body)),
      h("p", { class: "small", style: "margin:12px 0 0" }, "행을 누르면 비교 대상, 판정의 의미, 커널 구현을 볼 수 있습니다. 같은 내용은 ",
        h("a", { href: "api/registry", target: "_blank" }, "api/registry"), "에서 JSON으로 받을 수 있습니다.")));
  draw();
}
