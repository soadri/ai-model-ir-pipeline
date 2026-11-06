// 서버 API 호출. ?demo=preset|custom 으로 열면 저장된 샘플 결과로 화면만 확인할 수 있습니다.

const DEMO = new URLSearchParams(location.search).get("demo");

async function json(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function demo(name) {
  return json(`static/demo/${name}.json`);
}

export const api = {
  demo: DEMO,

  info() {
    if (DEMO) return Promise.resolve({ version: "0.1.1", kernel: { backend: "local", simulator: "furiosa-opt 0.6", prebuilt_kernels: 11 }, scales: ["sim_small", "sim_mid"] });
    return json("api/info");
  },

  cached(scale) {
    if (DEMO === "preset") return demo("preset").then((d) => ({ ...d, _from_cache: true }));
    if (DEMO) return Promise.resolve({ _from_cache: false });
    return json(`api/verify/cached_result?scale=${encodeURIComponent(scale)}`);
  },

  verify(scale) {
    if (DEMO) return demo("preset");
    return json(`api/verify/all_verify?scale=${encodeURIComponent(scale)}`);
  },

  convert(code) {
    if (DEMO) return demo("custom");
    return json("api/verify/custom_convert", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code }),
    });
  },

  runs() {
    if (DEMO) return Promise.resolve([]);
    return json("api/verify/ir/runs");
  },

  run(id) { return json(`api/verify/ir/runs/${encodeURIComponent(id)}`); },

  url: {
    rngd: (id) => `api/verify/ir/runs/${encodeURIComponent(id)}/rngd`,
    linalg: (id) => `api/verify/ir/runs/${encodeURIComponent(id)}/linalg`,
    kernel: (id, m) => `api/verify/ir/runs/${encodeURIComponent(id)}/kernels/${encodeURIComponent(m)}`,
  },
};
