// AI 모델 변환(표준 IR) 파이프라인 — 소개 페이지
// data/scenes.json : 첫 화면 변환 예시 (실제 파이프라인 출력에서 추출)
// data/status.json : 검증 기록 (GitHub Actions가 릴리즈·Upstream 확인 때 다시 만듦)

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// IR 문장 강조: linalg 연산, NPU dialect 연산, 핵심 속성
function irLine(text) {
  const out = h("span");
  const re = /(linalg\.\w+)|("rngd\.\w+")|((?:transpose_rhs|op) = "[^"]*")/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    out.append(text.slice(last, m.index));
    out.append(h("span", { class: m[1] ? "k-l" : m[2] ? "k-n" : "k-a" }, m[0]));
    last = m.index + m[0].length;
  }
  out.append(text.slice(last));
  return out;
}

const fmtDate = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}년 ${d.getMonth() + 1}월 ${d.getDate()}일 ${p(d.getHours())}:${p(d.getMinutes())}`;
};

// ── 명령 복사 ─────────────────────────────────────────────────────
for (const btn of document.querySelectorAll("[data-copy]")) {
  btn.addEventListener("click", async () => {
    const text = document.getElementById(btn.dataset.copy).textContent;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "복사됨";
      btn.dataset.done = "";
    } catch {
      btn.textContent = "직접 복사하세요";
    }
    setTimeout(() => { btn.textContent = "복사"; delete btn.dataset.done; }, 1800);
  });
}

// ── 변환 예시 ─────────────────────────────────────────────────────
const lens = {
  scenes: [], scene: null,
  tabs: $(".lens-tabs"), count: $(".lens-count"),
  pytorch: $("#lens-pytorch"), linalg: $("#lens-linalg"), npu: $("#lens-npu"),
  links: $("#lens-links"), pair: $(".pair"), detail: $("#lens-detail"),
  note: $("#lens-note"), source: $("#lens-source"),
};

const npuKind = (raw) => (raw.match(/"rngd\.(\w+)"/) || [])[1] || "";
const linalgKind = (raw) => (raw.match(/linalg\.(\w+)/) || [])[1] || "";
// 칩 보조 표기: 라벨과 같은 말은 반복하지 않음
const linalgTag = (op) => {
  if (op.kind === "absorbed") return "흡수";
  const k = linalgKind(op.raw);
  if (k === "generic") return op.kind === "reduction" ? "reduction" : "generic";
  return k === op.label ? "" : k;
};
const npuTag = (op) => (op.raw.includes("transpose_rhs") ? "transpose_rhs" : npuKind(op.raw));

const WHY = {
  absorbed: "출력 텐서를 초깃값으로 채우는 연산입니다. 뒤따르는 연산이 초깃값을 직접 다루므로 별도의 NPU 연산 없이 흡수됩니다.",
  transpose: "가중치의 축 순서를 바꾸는 linalg.generic입니다. (d0, d1) → (d1, d0) 형태의 indexing_map을 가지며, 뒤따르는 행렬 곱의 transpose_rhs 속성으로 합쳐집니다.",
  reduction: "한 축을 따라 값을 모으는 리덕션 linalg.generic입니다. 내부 연산에 따라 reduce의 종류(sum, max)가 정해집니다.",
};

function renderScene(key, animate = true) {
  const s = lens.scenes.find((x) => x.key === key) || lens.scenes[0];
  lens.scene = s;
  for (const b of lens.tabs.children) b.setAttribute("aria-selected", String(b.dataset.key === s.key));

  lens.pytorch.textContent = s.pytorch;
  lens.linalg.replaceChildren(...s.linalg.map((op, i) => h("button", {
    type: "button", class: `chip${op.kind === "absorbed" ? " absorbed" : ""}`, "data-side": "l", "data-i": i,
    "aria-label": `표준 IR 연산 ${op.label}`,
  }, op.label, op.kind === "absorbed" ? h("em", {}, "흡수") : null)));
  lens.npu.replaceChildren(...s.npu.map((op, j) => h("button", {
    type: "button", class: "chip", "data-side": "n", "data-j": j, "aria-label": `NPU 특화 IR 연산 ${op.label}`,
  }, op.label)));

  const nLinalg = s.linalg.length, nAbsorbed = s.linalg.filter((o) => o.kind === "absorbed").length;
  lens.count.replaceChildren(
    "표준 IR 연산 ", h("b", {}, nLinalg), "개 → NPU 특화 IR 연산 ", h("b", {}, s.npu.length), "개",
    nAbsorbed ? `, 흡수 ${nAbsorbed}개` : "",
    s.pattern ? `, 복합 패턴 ${s.pattern}` : "");
  lens.note.textContent = s.note;

  drawLinks(animate);
  const first = s.linalg.findIndex((o) => o.to !== null && o.to !== undefined);
  focusChip("l", first < 0 ? 0 : first);
}

function center(el, side) {
  const r = el.getBoundingClientRect(), p = lens.pair.getBoundingClientRect();
  return { x: r.left - p.left + r.width / 2, y: (side === "top" ? r.top : r.bottom) - p.top };
}

function drawLinks(animate) {
  const s = lens.scene;
  if (!s) return;
  const svg = lens.links;
  svg.replaceChildren();
  svg.classList.remove("drawing");
  const lChips = lens.linalg.children, nChips = lens.npu.children;
  s.linalg.forEach((op, i) => {
    if (op.to === null || op.to === undefined || !nChips[op.to]) return;
    const a = center(lChips[i], "bottom"), b = center(nChips[op.to], "top");
    const dy = Math.max(24, (b.y - a.y) * 0.55);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M${a.x} ${a.y} C${a.x} ${a.y + dy}, ${b.x} ${b.y - dy}, ${b.x} ${b.y}`);
    path.dataset.i = i;
    path.dataset.j = op.to;
    svg.append(path);
  });
  if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    for (const p of svg.children) p.style.setProperty("--len", Math.ceil(p.getTotalLength()));
    void svg.getBoundingClientRect();
    svg.classList.add("drawing");
  }
  markOn();
}

let focused = null;

function markOn() {
  if (!focused) return;
  const s = lens.scene;
  const { side, idx } = focused;
  const onL = new Set(), onN = new Set();
  if (side === "l") {
    onL.add(idx);
    const t = s.linalg[idx].to;
    if (t !== null && t !== undefined) { onN.add(t); s.linalg.forEach((o, i) => { if (o.to === t) onL.add(i); }); }
  } else {
    onN.add(idx);
    s.linalg.forEach((o, i) => { if (o.to === idx) onL.add(i); });
  }
  [...lens.linalg.children].forEach((c, i) => { c.classList.toggle("on", onL.has(i)); c.classList.toggle("dim", !onL.has(i)); });
  [...lens.npu.children].forEach((c, j) => { c.classList.toggle("on", onN.has(j)); c.classList.toggle("dim", !onN.has(j)); });
  for (const p of lens.links.children) p.classList.toggle("on", onL.has(+p.dataset.i) && onN.has(+p.dataset.j));
}

function focusChip(side, idx) {
  const s = lens.scene;
  focused = { side, idx };
  markOn();

  const rows = [];
  const add = (label, ...nodes) => rows.push(h("dt", {}, label), h("dd", {}, ...nodes));
  if (side === "l") {
    const op = s.linalg[idx];
    add("표준 IR", irLine(op.raw));
    if (op.body?.length) add("내부 연산", op.body.join(", "));
    else if (linalgTag(op) && op.kind !== "absorbed") add("종류", linalgTag(op));
    if (op.to !== null && op.to !== undefined) add("NPU 특화 IR", irLine(s.npu[op.to].raw));
    else add("NPU 특화 IR", h("span", { style: "color:var(--muted)" }, "없음 (흡수)"));
    if (WHY[op.kind]) rows.push(h("p", { class: "why" }, WHY[op.kind]));
  } else {
    const op = s.npu[idx];
    add("NPU 특화 IR", irLine(op.raw));
    add("종류", npuTag(op));
    const src = s.linalg.filter((o) => o.to === idx);
    src.forEach((o, k) => add(k === 0 ? "표준 IR" : "", irLine(o.raw)));
    if (src.length > 1) rows.push(h("p", { class: "why" }, `표준 IR 연산 ${src.length}개가 NPU 특화 IR 연산 하나로 결합됐습니다.`));
  }
  lens.detail.replaceChildren(h("dl", { style: "display:contents" }, rows));
}

function chipEvent(e) {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  if (chip.dataset.side === "l") focusChip("l", +chip.dataset.i);
  else focusChip("n", +chip.dataset.j);
}

async function initLens() {
  try {
    const data = await (await fetch("data/scenes.json")).json();
    lens.scenes = data.scenes;
    lens.source.textContent = `${data.source}. IR 문장은 indexing_maps 같은 속성과 결과 타입을 줄여 표시했습니다.`;
  } catch {
    $("#lens").hidden = true;
    return;
  }
  lens.tabs.replaceChildren(...lens.scenes.map((s) => h("button", {
    type: "button", role: "tab", "data-key": s.key, onclick: () => renderScene(s.key),
  }, s.title)));
  for (const box of [lens.linalg, lens.npu]) {
    box.addEventListener("mouseover", chipEvent);
    box.addEventListener("focusin", chipEvent);
    box.addEventListener("click", chipEvent);
  }
  renderScene(lens.scenes[0].key);
  let t;
  new ResizeObserver(() => { clearTimeout(t); t = setTimeout(() => drawLinks(false), 80); }).observe(lens.pair);
}

// ── 검증 기록 ─────────────────────────────────────────────────────
const SCALE_ROWS = [
  ["sim_small", "기본 모델", "Llama-3.1 8B Decoder Layer, dim 16"],
  ["sim_mid", "기본 모델, 중간 규모", "dim 64, attention head 8, KV head 2 (GQA)"],
  ["custom", "사용자 모델 경로", "SwiGLU FFN 예시 코드"],
];
const LIMIT = (name) => (/RMSNorm/.test(name) ? { pct: 1, kind: "elementwise" } : { pct: 5, kind: "contraction" });
const badge = (ok) => h("span", { class: ok ? "ok" : "ng" }, ok ? "통과" : "실패");
const pct = (v) => (v === null || v === undefined ? "–" : `${+(+v).toFixed(2)}%`);

function releaseBlock(rel) {
  const checks = rel.checks || {};
  const rows = SCALE_ROWS.filter(([k]) => checks[k]).map(([k, title, sub]) => {
    const c = checks[k];
    if (c.error) return h("tr", {}, h("td", {}, title, h("small", {}, sub)), h("td", { colspan: 5 }, "결과를 읽지 못했습니다"));
    const l12 = Array.isArray(c.l12) ? `${c.l12[0]} / ${c.l12[1]}` : (c.l12 ? "통과" : "실패");
    return h("tr", {},
      h("td", {}, title, h("small", {}, sub)),
      h("td", { class: "n" }, pct(c.coverage)),
      h("td", { class: "n" }, c.npu_ops ?? "–"),
      h("td", { class: "n" }, l12),
      h("td", { class: "n" }, `${c.l3[0]} / ${c.l3[1]}`),
      h("td", {}, badge(c.ok)));
  });
  const allOk = Object.values(checks).every((c) => c.ok);
  const mid = checks.sim_mid || checks.sim_small;

  return h("div", { class: "release" },
    h("div", { class: "rel-head" },
      h("h3", {}, rel.tag), badge(allOk),
      h("span", { class: "meta" }, `${fmtDate(rel.published)} 배포`),
      h("a", { href: rel.url }, "릴리즈 노트"),
      h("a", { href: "https://github.com/soadri/ai-model-ir-pipeline/pkgs/container/ai-model-ir-pipeline" }, "이미지")),
    h("div", { class: "table-scroll" }, h("table", { class: "ledger" },
      h("thead", {}, h("tr", {}, ["검증", "커버리지", "NPU dialect 연산", "Level 1-2", "Level 3", "결과"].map((t) => h("th", {}, t)))),
      h("tbody", {}, rows))),
    mid?.modules?.length ? h("div", { class: "tol" },
      h("h4", {}, `Level 3 상대 오차와 허용 기준 (${checks.sim_mid ? "중간 규모" : "기본 규모"})`),
      h("p", {}, "NPU 시뮬레이터에서 실행한 모듈별 결과를 PyTorch 기준값과 비교한 상대 오차입니다. 막대 끝이 연산 계열별 허용 기준(elementwise 1%, contraction 5%)입니다."),
      h("div", { class: "tol-rows" },
        mid.modules.map((m) => {
          const lim = LIMIT(m.name);
          const w = Math.min(100, ((m.rel || 0) / lim.pct) * 100);
          return h("div", { class: "tol-row" },
            h("span", { class: "name" }, m.name, h("small", {}, `${lim.kind}, 기준 ${lim.pct}%`)),
            h("span", { class: "bar", role: "img", "aria-label": `${m.name} 상대 오차 ${pct(m.rel)}, 허용 기준 ${lim.pct}%` },
              h("span", { class: lim.kind, style: `width:${Math.max(w, m.rel ? 1.5 : 0)}%` })),
            h("span", { class: "v" }, pct(m.rel)));
        }),
        h("div", { class: "tol-scale" }, h("span"), h("span", {}, h("span", {}, "0"), h("span", {}, "허용 기준")), h("span")))) : null,
    rel.digest ? h("p", { class: "stamp" }, "이미지 다이제스트 ", h("code", {}, rel.digest)) : null);
}

function upstreamBlock(up) {
  return h("div", { class: "up" },
    h("div", {},
      h("h4", {}, "고정 의존성"),
      h("p", {}, "체험 이미지를 다시 빌드하는 데 필요한 의존성을 지금도 받을 수 있는지 확인합니다."),
      h("div", { class: "table-scroll" }, h("table", { class: "ledger" },
        h("thead", {}, h("tr", {}, h("th", {}, "항목"), h("th", {}, "상태"))),
        h("tbody", {}, up.pinned.map((r) => h("tr", {},
          h("td", {}, r.name, h("small", {}, r.note)),
          h("td", { class: "st" }, h("span", { class: r.ok ? "ok" : "ng" }, r.ok ? "정상" : "문제")))))))),
    h("div", {},
      h("h4", {}, "Upstream 최신 버전"),
      h("p", {}, "파이프라인이 고정해 쓰는 버전과 Upstream의 최신 버전입니다. 버전이 바뀌면 같은 검증 기준으로 호환성을 확인한 뒤 갱신을 검토합니다."),
      h("div", { class: "table-scroll" }, h("table", { class: "ledger" },
        h("thead", {}, h("tr", {}, h("th", {}, "구성 요소"), h("th", {}, "고정"), h("th", {}, "최신"))),
        h("tbody", {}, up.latest.map((r) => h("tr", {},
          h("td", { style: "font-family:var(--sans);font-size:15px" }, r.name),
          h("td", { class: "ver" }, r.pinned || "–"),
          h("td", { class: `ver${r.pinned && r.latest !== r.pinned ? " newer" : ""}` }, r.latest))))))));
}

async function initRecord() {
  const body = $("#record-body");
  let st;
  try {
    st = await (await fetch("data/status.json", { cache: "no-cache" })).json();
  } catch {
    body.replaceChildren(h("p", { class: "loading" }, "검증 기록을 불러오지 못했습니다. ",
      h("a", { href: "https://github.com/soadri/ai-model-ir-pipeline/releases" }, "릴리즈 페이지"), "에서 확인하세요."));
    return;
  }
  const [latest, ...older] = st.releases || [];
  body.replaceChildren(...[
    latest ? releaseBlock(latest) : h("p", { class: "loading" }, "아직 릴리즈가 없습니다."),
    older.length ? h("details", { class: "older" }, h("summary", {}, `이전 릴리즈 ${older.length}개`), older.map(releaseBlock)) : null,
    st.upstream ? upstreamBlock(st.upstream) : null,
    h("p", { class: "stamp" },
      `마지막 갱신 ${fmtDate(st.generated_at)}. `,
      h("a", { href: st.run_url || `${st.repository}/actions/workflows/pages.yml` }, "이 기록을 만든 실행"),
      " 또는 ",
      h("a", { href: `${st.repository}/issues?q=label%3Aupstream` }, "Upstream 변화 감지 Issue"),
      "에서 자세한 내용을 볼 수 있습니다."),
  ].filter(Boolean));
}

initLens();
initRecord();
