// 결과 화면 — 기본 모델과 사용자 모델 응답을 같은 모양으로 정리한 뒤 그립니다.
import { api } from "../api.js";
import { append, badge, clear, download, fmt, h, icon } from "../dom.js";
import { MARKERS, PATTERNS, SCALES, STRATEGY } from "../labels.js";

// ── 표기 정리 ──────────────────────────────────────────────────────
// 서버가 돌려주는 내부 이름을 화면용 이름으로 바꿉니다 (식별자 rngd.* 는 IR 안에서만 그대로 표시).
const CHECK_NAMES = {
  "torch_mlir 컴파일": "torch-mlir 컴파일",
  "rewrite_to_rngd 실행": "NPU 특화 IR 변환",
  "rngd.* op 생성": "NPU dialect 연산 생성",
  "rngd.* op 출력 shape 유효성": "NPU dialect 연산 출력 shape",
  "미분류(UNKNOWN) op 비율": "미분류 연산 비율",
};
const tidy = (s) => String(s ?? "")
  .replace(/rngd\.\* op/g, "NPU dialect 연산")
  .replace(/\s*\(\{.*\}\)\s*$/, "")
  .replace(/\s*\(max_diff=[^)]*\)\s*$/, "");

const REASONS = {
  "단독 잔류 arith/math op": "다른 연산과 결합되지 못하고 단독으로 남은 연산입니다.",
};

const filterDense = (ir) => String(ir || "").split("\n")
  .map((l) => (l.includes("dense<") ? l.slice(0, l.indexOf("dense<")) + "dense<…>" : l)).join("\n");

function countOps(lines) {
  const counts = {};
  for (const line of lines || []) {
    const m = line.match(/"?rngd\.(\w+)"?/);
    if (m) counts[m[1]] = (counts[m[1]] || 0) + 1;
  }
  return Object.entries(counts).sort((a, b) => b[1] - a[1]);
}

// ── 응답 정리 ──────────────────────────────────────────────────────
export function normalizePreset(d, requested) {
  const l12 = d.level12 || {}, l3 = d.level3 || {}, ir = d.ir_detail || {};
  const scale = SCALES[l3.profile] ? l3.profile : requested;
  return {
    kind: "preset",
    title: "Llama-3.1 8B Decoder Layer",
    subtitle: `${SCALES[scale]?.title || scale} (${SCALES[scale]?.detail || ""})`,
    cachedAt: d._from_cache ? d._cached_at : null,
    coverage: ir.coverage_pct ?? l3.ir?.coverage_pct,
    npuOps: ir.rngd_count ?? l12.rngd_count,
    pytorch: [["출력", `[${(l12.ref_output_shape || []).join(", ")}]`], ["자료형", (l12.ref_output_dtype || "").replace("torch.", "")],
              ["모듈", `${(l3.captures || []).length}개`]],
    linalg: { ops: l12.ops_before, lines: ir.linalg_lines, markers: ir.markers_summary || l12.markers || {} },
    npu: { ops: countOps(ir.rngd_ops), fusion: l12.fusion || {}, edges: l12.dataflow_edges },
    unsupported: [],
    l12: {
      pass: l12.pass_count, total: (l12.checks || []).length, ok: l12.all_passed,
      checks: (l12.checks || []).map((c) => ({ name: CHECK_NAMES[c.name] || tidy(c.name), ok: c.passed, value: tidy(c.actual), expected: tidy(c.expected) })),
    },
    l3: {
      pass: l3.pass_count, total: l3.total, ok: l3.all_passed, error: l3.error,
      rows: (l3.checks || []).map((c) => ({
        name: c.module, chain: String(c.detail || "").split(". ")[0], kernel: c.e2e_bin,
        diff: c.max_diff, rel: c.rel_err_pct, judgment: c.judgment, ok: c.passed,
      })),
    },
    ir: { linalg: filterDense(ir.linalg_ir_full || ir.linalg_preview), npu: ir.rngd_ir_full || filterDense(ir.rngd_preview),
          npuOps: ir.rngd_ops || [] },
    raw: d,
  };
}

export function normalizeCustom(d) {
  const l12 = d.level12 || {}, l3 = d.level3 || {}, s = l3.summary || {};
  return {
    kind: "custom",
    title: "사용자 정의 모델",
    subtitle: `소요 시간 ${d.elapsed_s ?? "–"}초`,
    coverage: d.coverage_pct,
    npuOps: d.converted_count,
    pytorch: [["출력", l12.output_shape ? `[${l12.output_shape.join(", ")}]` : "–"]],
    linalg: { lines: d.ir?.linalg_lines, markers: d.markers || {} },
    npu: { ops: countOps(d.ir?.rngd_ops), fusion: d.fusion || {} },
    unsupported: d.unsupported || [],
    l12custom: l12,
    l3: {
      pass: s.passed ?? 0, total: s.tested ?? 0, ok: s.all_passed, error: l3.error,
      layers: (l3.results || []).map((r) => ({
        name: r.layer, shape: r.type ? `${r.type} (${r.shape || ""})` : r.shape,
        kind: (r.op_type || "").replace(/^.*\((.*)\)$/, "$1"), strategy: r.strategy, reason: r.reason,
        ok: r.npu_result?.passed, detail: r.npu_result?.detail, tiling: r.npu_result?.tiling_metrics,
        tests: r.npu_result?.test_results,
      })),
    },
    refs: d.npu_ref,
    ir: { linalg: d.ir?.linalg_ir_full || filterDense(d.ir?.linalg_preview),
          npu: d.ir?.rngd_ir_full || filterDense(d.ir?.rngd_preview), npuOps: d.ir?.rngd_ops || [] },
    raw: d,
  };
}

// ── 화면 조각 ──────────────────────────────────────────────────────
function summary(v) {
  const covCls = v.coverage >= 100 ? "pass" : v.coverage >= 70 ? "warn" : "fail";
  const cell = (k, value, sub, cls = "") => h("div", {}, h("div", { class: "k" }, k),
    h("div", { class: `v ${cls}` }, value, sub ? h("small", {}, sub) : null));
  const l12 = v.kind === "preset"
    ? cell("Level 1-2 IR 정합성", `${v.l12.pass}`, `/ ${v.l12.total}`, v.l12.ok ? "pass" : "fail")
    : cell("Level 1-2 수치 비교", v.l12custom.passed ? "통과" : "실패", null, v.l12custom.passed ? "pass" : "fail");
  return h("div", { class: "summary" },
    cell("변환 커버리지", fmt.pct(v.coverage), null, covCls),
    cell("NPU dialect 연산", `${v.npuOps ?? "–"}`, "개"),
    l12,
    cell("Level 3 NPU 시뮬레이터", `${v.l3.pass}`, `/ ${v.l3.total}`, v.l3.ok ? "pass" : "fail"));
}

function markerBar(markers) {
  const items = MARKERS.filter((m) => markers[m.key]);
  const total = items.reduce((s, m) => s + markers[m.key], 0) || 1;
  return [
    h("div", { class: "bar", role: "img", "aria-label": "연산 분류 비율" },
      items.map((m) => h("span", { class: m.cls, style: `width:${(markers[m.key] / total) * 100}%`, title: `${m.label} ${markers[m.key]}` }))),
    h("div", { class: "legend" }, items.map((m) => h("span", {}, h("i", { class: m.cls }), m.label, h("b", {}, markers[m.key])))),
  ];
}

function strata(v) {
  const facts = (pairs) => h("div", { class: "facts" }, pairs.filter(([, x]) => x !== undefined && x !== null && x !== "")
    .map(([k, x]) => h("span", {}, `${k} `, h("b", {}, x))));
  const fusion = Object.entries(v.npu.fusion);
  return h("div", { class: "strata" },
    h("div", { class: "stratum slate" },
      h("div", { class: "name" }, "PyTorch", h("small", {}, "nn.Module")),
      h("div", { class: "body" }, facts(v.pytorch))),
    h("div", { class: "stratum blue" },
      h("div", { class: "name" }, "표준 MLIR", h("small", {}, "linalg dialect")),
      h("div", { class: "body" },
        facts([["linalg IR", v.linalg.lines ? `${v.linalg.lines}줄` : null], ["변환 전 연산", v.linalg.ops ? `${v.linalg.ops}개` : null]]),
        markerBar(v.linalg.markers))),
    h("div", { class: "stratum violet" },
      h("div", { class: "name" }, "NPU 특화 IR", h("small", {}, "NPU dialect")),
      h("div", { class: "body" },
        h("div", { class: "ops" }, v.npu.ops.map(([op, n]) => h("span", { class: "op" }, op, h("em", {}, `×${n}`)))),
        fusion.length ? h("div", { class: "ops" }, h("span", { class: "small", style: "align-self:center;margin-right:4px" }, "복합 패턴"),
          fusion.map(([k, n]) => h("span", { class: "pattern" }, PATTERNS[k] || k, h("em", {}, `×${n}`)))) : null)));
}

function unsupported(v) {
  if (!v.unsupported.length) return null;
  return h("section", { class: "section" },
    h("header", {}, h("h2", {}, "아직 변환하지 못한 연산"),
      h("p", { class: "hint" }, "변환 규칙이 없어 CPU에서 처리되는 연산입니다. 이유와 구성 연산을 함께 보여드립니다.")),
    h("div", { class: "unsupported" }, v.unsupported.map((u) => h("article", {},
      icon("warn"),
      h("h3", {}, u.type === "group" ? `${u.label} (복합 연산)` : u.name),
      h("p", {}, u.type === "group" ? [u.reason, " 구성: ", h("code", {}, (u.body || []).join(", "))] : (REASONS[u.reason] || u.reason))))));
}

function level12Card(v) {
  if (v.kind === "custom") {
    const r = v.l12custom;
    return h("div", { class: "card" },
      h("header", {}, h("h3", {}, "Level 1-2", h("small", {}, "표준 IR 수치 비교")), badge(r.passed)),
      h("p", { class: "hint", style: "margin-bottom:12px" }, "PyTorch 출력과 linalg IR을 RefBackend로 실행한 결과를 비교합니다."),
      r.error ? h("div", { class: "error" }, r.error) : h("dl", { class: "kv" },
        h("dt", {}, "상대 오차"), h("dd", {}, fmt.pct(r.rel_err_pct)),
        h("dt", {}, "최대 절대 오차"), h("dd", {}, fmt.num(r.max_diff)),
        h("dt", {}, "평균 절대 오차"), h("dd", {}, fmt.num(r.mean_diff)),
        h("dt", {}, "출력 shape"), h("dd", {}, `[${(r.output_shape || []).join(", ")}]`)));
  }
  return h("div", { class: "card" },
    h("header", {}, h("h3", {}, "Level 1-2", h("small", {}, "IR 변환 정합성")), badge(v.l12.ok, `${v.l12.pass} / ${v.l12.total} 통과`)),
    h("ul", { class: "checks" }, v.l12.checks.map((c) => h("li", { title: c.expected ? `기준: ${c.expected}` : null },
      icon(c.ok ? "check" : "cross", c.ok ? "ok" : "ng"), h("span", {}, c.name), h("span", { class: "val" }, c.value)))));
}

function level3Card(v) {
  const head = h("header", {}, h("h3", {}, "Level 3", h("small", {}, "NPU 시뮬레이터(Functional) 연산 정합성")),
    badge(v.l3.ok, `${v.l3.pass} / ${v.l3.total} 통과`));
  if (v.l3.error) return h("div", { class: "card" }, head, h("div", { class: "error" }, v.l3.error));

  if (v.kind === "preset") {
    return h("div", { class: "card" }, head,
      h("div", { class: "table-wrap" }, h("table", { class: "t" },
        h("thead", {}, h("tr", {}, h("th", {}, "모듈"), h("th", { style: "text-align:right" }, "오차"), h("th", {}, "판정"))),
        h("tbody", {}, v.l3.rows.map((r) => h("tr", { title: r.judgment || null },
          h("td", {}, h("b", {}, r.name), h("span", { class: "small" }, r.chain), h("span", { class: "small mono" }, r.kernel)),
          h("td", { class: "num" }, r.rel !== undefined && r.rel !== null ? [`상대 ${fmt.pct(r.rel)}`, h("span", { class: "small" }, `최대 ${fmt.num(r.diff)}`)] : fmt.num(r.diff)),
          h("td", {}, badge(r.ok))))))),
      h("p", { class: "small", style: "margin:10px 0 0" },
        "오차는 연산 계열별 허용 기준(상대 오차 elementwise 1%, contraction 5%)으로 판정합니다. 행렬 곱 모듈은 출력값 규모가 커서 절대 오차가 크게 보일 수 있습니다."));
  }

  return h("div", { class: "card" }, head,
    h("div", { class: "table-wrap" }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, h("th", {}, "레이어"), h("th", {}, "검증 방식"), h("th", {}, "결과"))),
      h("tbody", {}, v.l3.layers.map((r) => h("tr", {},
        h("td", {}, h("b", {}, r.name), h("span", { class: "small" }, r.shape), r.kind ? h("span", { class: "small mono" }, r.kind) : null),
        h("td", {}, STRATEGY[r.strategy] || "그대로 실행", r.reason ? h("span", { class: "small" }, r.reason) : null,
          r.tiling?.accumulation ? h("span", { class: "small" }, `누적 상대 오차 ${fmt.pct(r.tiling.accumulation.rel_err_pct)}`) : null),
        h("td", {}, badge(r.ok))))))),
    v.l3.layers.length ? null : h("p", { class: "hint" }, "시뮬레이터로 검증할 수 있는 레이어(Linear, Norm, SiLU)가 없습니다."));
}

function refsCard(v) {
  if (!v.refs?.refs?.length) return null;
  return h("div", { class: "card" },
    h("header", {}, h("h3", {}, "연산별 검증 근거", h("small", {}, v.refs.basis)),
      v.refs.all_verified ? h("span", { class: "badge pass" }, icon("check"), "모두 검증됨") : h("span", { class: "badge warn" }, icon("warn"), "일부 미검증")),
    h("div", { class: "table-wrap" }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, h("th", {}, "NPU dialect 연산"), h("th", { style: "text-align:right" }, "사용"), h("th", { style: "text-align:right" }, "상대 오차"), h("th", {}, "근거"))),
      h("tbody", {}, v.refs.refs.map((r) => h("tr", {},
        h("td", {}, h("code", {}, r.op.replace(/^rngd\./, ""))),
        h("td", { class: "num" }, `${r.count}회`),
        h("td", { class: "num" }, r.verified ? fmt.pct(r.rel_err_pct) : "–"),
        h("td", {}, r.verified ? r.source : h("span", { class: "badge warn" }, "검증 예정"))))))));
}

function verifySection(v) {
  return h("section", { class: "section" },
    h("header", {}, h("h2", {}, "검증")),
    h("div", { class: "verify" }, level12Card(v), level3Card(v)),
    refsCard(v),
    h("div", { class: "note" }, icon("info"),
      h("span", {}, "Level 3는 NPU 시뮬레이터(Functional)에서 연산 결과의 정합성을 확인하며, 실제 NPU의 실행 성능을 나타내지 않습니다.",
        v.kind === "preset" ? " 기본 모델은 같은 조건(seed 42)의 PyTorch 모델에서 미리 만든 기준값과 비교합니다." : "")));
}

function highlight(text) {
  const pre = h("pre", { tabindex: "0" });
  const re = /("?(?:linalg|rngd)\.[\w.]+"?)|(tensor<[^>]*>)|(\b(?:transpose_rhs|op|in_shape|out_shape)\b\s*=)/g;
  const s = text || "(표시할 IR이 없습니다)";
  let last = 0, m;
  while ((m = re.exec(s))) {
    pre.append(s.slice(last, m.index));
    const cls = m[1] ? (m[1].includes("rngd.") ? "k-rngd" : "k-linalg") : m[2] ? "k-type" : "k-attr";
    pre.append(h("span", { class: cls }, m[0]));
    last = m.index + m[0].length;
  }
  pre.append(s.slice(last));
  return pre;
}

function irSection(v) {
  const tabs = [["npu", "NPU 특화 IR"], ["linalg", "표준 IR (linalg)"]];
  if (v.ir.npuOps?.length) tabs.unshift(["ops", "NPU 연산 목록"]);
  const box = h("div", { class: "ir" });
  const bar = h("div", { class: "ir-bar", role: "tablist" });
  const body = h("div", {});
  const show = (key) => {
    for (const b of bar.querySelectorAll("button")) b.setAttribute("aria-selected", String(b.dataset.key === key));
    const text = key === "ops" ? v.ir.npuOps.join("\n") : v.ir[key];
    clear(body).append(highlight(text));
    info.textContent = `${(text || "").split("\n").length}줄`;
  };
  for (const [key, label] of tabs) bar.append(h("button", { role: "tab", dataset: { key }, onclick: () => show(key) }, label));
  const info = h("span", { class: "small" });
  bar.append(h("span", { class: "spacer" }), info);
  box.append(bar, body);
  show(tabs[0][0]);
  return h("section", { class: "section" },
    h("header", {}, h("h2", {}, "IR 보기"), h("p", { class: "hint" }, "변환된 연산 목록과 IR 전체를 볼 수 있습니다. 가중치 상수는 생략했습니다.")),
    box);
}

function downloadsSection(v, runId) {
  const items = [h("button", { class: "dl-btn", onclick: () => download(`report_${v.kind}_${Date.now()}.json`, JSON.stringify(v.raw, null, 2)) },
    icon("down"), "검증 리포트", h("small", {}, "JSON"))];
  const slot = h("div", { class: "downloads" }, items);
  if (runId) {
    slot.append(
      h("a", { class: "dl-btn", href: api.url.rngd(runId) }, icon("file"), "NPU 특화 IR", h("small", {}, ".mlir")),
      h("a", { class: "dl-btn", href: api.url.linalg(runId) }, icon("file"), "표준 IR", h("small", {}, ".mlir")));
    api.run(runId).then((detail) => {
      for (const k of detail.kernels || []) {
        slot.append(h("a", { class: "dl-btn", href: api.url.kernel(runId, k.module_name) }, icon("file"), `${k.module_name} 커널`, h("small", {}, ".rs")));
      }
    }).catch(() => {});
  }
  return h("section", { class: "section" }, h("header", {}, h("h2", {}, "내려받기")), slot);
}

// ── 전체 ───────────────────────────────────────────────────────────
export function renderResult(root, v, runId) {
  clear(root);
  append(root, [
    h("div", { class: "result-head" },
      h("div", {}, h("h1", {}, v.title), h("div", { class: "meta" }, v.subtitle)),
      v.cachedAt ? h("span", { class: "cached" }, `이전 실행 결과 ${fmt.time(v.cachedAt)}`) : null),
    summary(v),
    h("section", { class: "section" },
      h("header", {}, h("h2", {}, "변환 단계"), h("p", { class: "hint" }, "모델이 표준 MLIR을 거쳐 NPU 특화 IR로 낮아지는 과정입니다.")),
      strata(v)),
    unsupported(v),
    verifySection(v),
    irSection(v),
    downloadsSection(v, runId),
  ]);
}
