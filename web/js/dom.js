// 작은 DOM 헬퍼 — 문자열 결합(innerHTML) 대신 요소를 직접 만들어 내용이 안전하게 들어가도록 합니다.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) { el.replaceChildren(); return el; }

const SVG_NS = "http://www.w3.org/2000/svg";
const ICONS = {
  check: "M3.5 8.5l3 3 6-7",
  cross: "M4.5 4.5l7 7m0-7l-7 7",
  warn: "M8 2.5l6 11H2l6-11zM8 6.5v3.2M8 11.6v.1",
  info: "M8 14A6 6 0 108 2a6 6 0 000 12zM8 7.2v4M8 5v.1",
  down: "M8 2.5v8m-3.5-3.5L8 10.5l3.5-3.5M3 13.5h10",
  file: "M4 1.8h5l3 3v9.4H4zM9 1.8v3h3",
};

export function icon(name, cls = "") {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("class", `ico ${cls}`.trim());
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(SVG_NS, "path");
  p.setAttribute("d", ICONS[name]);
  p.setAttribute("fill", "none");
  p.setAttribute("stroke", "currentColor");
  p.setAttribute("stroke-width", "1.6");
  p.setAttribute("stroke-linecap", "round");
  p.setAttribute("stroke-linejoin", "round");
  svg.append(p);
  return svg;
}

export function badge(ok, okText = "통과", ngText = "실패") {
  return h("span", { class: `badge ${ok ? "pass" : "fail"}` }, icon(ok ? "check" : "cross"), ok ? okText : ngText);
}

export const fmt = {
  num(v, digits = 4) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a >= 1e-3) return v.toFixed(digits);
    return v.toExponential(2);
  },
  pct(v) {
    if (v === null || v === undefined || Number.isNaN(+v)) return "–";
    const n = +v;
    if (n === 0) return "0%";
    if (Math.abs(n) < 0.01) return `${n.toExponential(1)}%`;
    return `${+n.toFixed(2)}%`;
  },
  time(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(+d)) return iso;
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getMonth() + 1}월 ${d.getDate()}일 ${p(d.getHours())}:${p(d.getMinutes())}`;
  },
};

export function download(filename, text, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
