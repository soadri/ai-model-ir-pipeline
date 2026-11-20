"""
예시 기반 패턴 식별 엔진 (그래프 템플릿 매칭)

GEP-0034 「AI 모델 중간 표현의 연산 패턴 식별을 위한 예시 기반 분석 기법」의 구현을 공개판으로 옮긴 것.
라벨을 붙인 예시 노드로 패턴 템플릿을 만들고(extract_pattern_from_example), 데이터 흐름 그래프에서
유사도(세부 연산 일치 1.0, 연산 종류만 일치 0.5)로 같은 모양을 찾는다(PatternLibrary.find_all).

규칙 기반 식별(e2e_pipeline.detect_fusion_patterns)을 대체하지 않고 나란히 실행한다.
변환·검증·최적화 마커는 계속 규칙 기반 결과를 쓴다.

연구용 구현(v3b) 대비 수정:
  - 다음 노드 후보를 가까운 단계부터 확인 (직접 연결 우선, 동점이면 위치가 가까운 노드)
  - 보조 경로(직접 연결이 끊긴 경우 앞뒤 5노드 탐색)도 가까운 노드부터 확인
  - 맥락 패턴(match_mode="context", Residual)은 맥락 단계에서만 찾고, 맥락 조건은
    「함수 인자 입력 또는 5노드 이상 앞에서 온 입력(Skip Connection)」
  - 같은 템플릿의 후보가 겹치면 노드가 촘촘하게 모인 후보를 먼저 채택
  - 템플릿에 종류(kind)와 변형(variant)을 두고, 같은 종류끼리 노드가 절반 이상 겹치면 큰 매칭만 남김
  - 노드 공유 금지는 같은 종류끼리만 (FFN 안의 SiLU·GELU처럼 종류가 다른 패턴은 겹칠 수 있음)
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import json


# =====================================================================
# 데이터 구조
# =====================================================================

@dataclass
class PatternNode:
    id: int; op_type: str; op_attr: str
    successors: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)
    context: dict = field(default_factory=dict)

@dataclass
class PatternGraph:
    name: str; nodes: list; description: str = ""
    is_chain: bool = False; sub_patterns: list = field(default_factory=list)
    kind: str = ""          # 결과에 쓰는 패턴 이름 (예: FFN). 비어 있으면 name
    variant: str = ""       # 같은 종류 안의 변형 (예: GeGLU)
    source: dict = field(default_factory=dict)   # 템플릿을 만든 예시 (모델, 라벨 노드)
    match_mode: str = ""    # "context": 모양만으로는 구분되지 않아 맥락 조건으로만 찾는 패턴 (Residual)
    @property
    def size(self): return len(self.nodes)
    @property
    def entry_nodes(self): return [n for n in self.nodes if not n.predecessors]
    @property
    def exit_nodes(self): return [n for n in self.nodes if not n.successors]
    def to_dict(self):
        return {"name":self.name,"kind":self.kind or self.name,"variant":self.variant,"source":self.source,"match_mode":self.match_mode,
                "description":self.description,"is_chain":self.is_chain,
                "sub_patterns":self.sub_patterns,
                "nodes":[{"id":n.id,"op_type":n.op_type,"op_attr":n.op_attr,
                          "successors":n.successors,"predecessors":n.predecessors,
                          "context":n.context} for n in self.nodes]}
    @classmethod
    def from_dict(cls, d):
        ns = [PatternNode(id=n["id"],op_type=n["op_type"],op_attr=n["op_attr"],
                          successors=n.get("successors",[]),predecessors=n.get("predecessors",[]),
                          context=n.get("context",{})) for n in d["nodes"]]
        return cls(name=d["name"],nodes=ns,description=d.get("description",""),
                   is_chain=d.get("is_chain",False),sub_patterns=d.get("sub_patterns",[]),
                   kind=d.get("kind",""),variant=d.get("variant",""),source=d.get("source",{}),
                   match_mode=d.get("match_mode",""))

@dataclass
class MatchResult:
    pattern_name: str; matched_indices: list; similarity: float
    details: dict = field(default_factory=dict)


# =====================================================================
# 그래프 유틸리티
# =====================================================================

def _build_adjacency(nodes):
    ssa_to_idx = {}
    for n in nodes:
        if n.ssa_name: ssa_to_idx[n.ssa_name] = n.idx
    succ = {n.idx:[] for n in nodes}
    pred = {n.idx:[] for n in nodes}
    for n in nodes:
        for inp in n.inputs:
            pi = ssa_to_idx.get(inp, -1)
            if pi >= 0 and pi in succ:
                if n.idx not in succ[pi]: succ[pi].append(n.idx)
                if pi not in pred[n.idx]: pred[n.idx].append(pi)
    return ssa_to_idx, succ, pred

def _node_match_score(pn, dn, strict=True):
    if pn.op_type=="*" and pn.op_attr=="*": return 0.8
    tm = pn.op_type=="*" or pn.op_type==dn.op_type
    if pn.op_attr=="" and dn.op_attr=="": am = True
    else: am = pn.op_attr=="*" or pn.op_attr==dn.op_attr
    if not strict: return 1.0 if am else 0.0
    if tm and am: return 1.0
    elif tm: return 0.5
    return 0.0


# =====================================================================
# 예시 기반 패턴 템플릿 생성 (v3b: 직접 연결만 + 고립 노드 복구)
# =====================================================================

def extract_pattern_from_example(nodes, labeled_indices, pattern_name, description=""):
    idx_to_node = {n.idx: n for n in nodes}
    labeled_set = set(labeled_indices)
    sorted_idx = sorted(labeled_indices)
    idx_to_pid = {idx: i for i, idx in enumerate(sorted_idx)}

    ssa_to_idx = {}
    for n in nodes:
        if n.ssa_name: ssa_to_idx[n.ssa_name] = n.idx

    # Step 1: 직접 SSA 연결만으로 패턴 구성
    pattern_nodes = []
    for idx in sorted_idx:
        dn = idx_to_node[idx]
        successors = []
        predecessors = []

        # 직접 연결: 이 노드의 출력 SSA를 입력으로 사용하는 라벨 노드
        for other_idx in sorted_idx:
            if other_idx == idx: continue
            other_dn = idx_to_node[other_idx]
            if dn.ssa_name in other_dn.inputs:
                if other_idx > idx:
                    successors.append(idx_to_pid[other_idx])
                else:
                    pass  # 역방향은 predecessors에서 처리

        # 직접 연결: 이 노드의 입력 SSA가 라벨 노드의 출력인 경우
        for inp_ssa in dn.inputs:
            src_idx = ssa_to_idx.get(inp_ssa, -1)
            if src_idx in labeled_set and src_idx != idx:
                predecessors.append(idx_to_pid[src_idx])

        # 컨텍스트
        ext = []
        if hasattr(dn, 'input_sources') and dn.input_sources:
            for inp, src in zip(dn.inputs, dn.input_sources):
                si = ssa_to_idx.get(inp, -1)
                if si not in labeled_set: ext.append(src)
        ctx = {}
        if ext: ctx["external_sources"] = ext
        if hasattr(dn, 'input_sources') and dn.input_sources:
            ctx["input_sources"] = list(dn.input_sources)

        pn = PatternNode(id=idx_to_pid[idx], op_type=dn.op_type, op_attr=dn.op_attr,
                         successors=sorted(set(successors)),
                         predecessors=sorted(set(predecessors)), context=ctx)
        pattern_nodes.append(pn)

    # Step 2: 고립 노드 복구 — 토폴로지 순서로 가장 가까운 이웃 연결
    for i, pn in enumerate(pattern_nodes):
        if pn.successors or pn.predecessors:
            continue  # 이미 연결됨
        if len(pattern_nodes) <= 1:
            continue

        # 가장 가까운 후속 연결 노드 찾기 (토폴로지 순서)
        best_succ = None
        for j in range(i+1, len(pattern_nodes)):
            candidate = pattern_nodes[j]
            # 연결된 노드이거나 진입 노드면 연결 대상
            if candidate.predecessors or candidate.successors or j == i+1:
                best_succ = j
                break

        # 가장 가까운 선행 연결 노드 찾기
        best_pred = None
        for j in range(i-1, -1, -1):
            candidate = pattern_nodes[j]
            if candidate.predecessors or candidate.successors or j == i-1:
                best_pred = j
                break

        # 연결 추가 (가까운 쪽 우선)
        if best_succ is not None:
            if best_succ not in pn.successors:
                pn.successors.append(best_succ)
            if pn.id not in pattern_nodes[best_succ].predecessors:
                pattern_nodes[best_succ].predecessors.append(pn.id)

        if best_pred is not None:
            if best_pred not in pn.predecessors:
                pn.predecessors.append(best_pred)
            if pn.id not in pattern_nodes[best_pred].successors:
                pattern_nodes[best_pred].successors.append(pn.id)

    # is_chain 판단
    has_isolated = any(not n.successors and not n.predecessors for n in pattern_nodes if len(pattern_nodes)>1)
    linear = all(len(n.successors)<=1 and len(n.predecessors)<=1 for n in pattern_nodes)
    is_chain = linear and not has_isolated

    return PatternGraph(name=pattern_name, nodes=pattern_nodes,
                        description=description, is_chain=is_chain)


# =====================================================================
# DAG 매칭 엔진
# =====================================================================

def _match_dag(pattern, nodes, anchor, succ_map, pred_map, idx_to_node, threshold, strict):
    mapping = dict(anchor)
    scores = []
    for pid, di in mapping.items():
        s = _node_match_score(pattern.nodes[pid], idx_to_node[di], strict)
        if s < threshold: return None
        scores.append(s)

    visited = set(mapping.keys())
    rev = {v:k for k,v in mapping.items()}
    queue = deque()
    for pid in list(mapping.keys()):
        pn = pattern.nodes[pid]
        for sp in pn.successors:
            if sp not in visited: queue.append(("fwd", pid, sp))
        for pp in pn.predecessors:
            if pp not in visited: queue.append(("bwd", pid, pp))

    while queue:
        d, ref_pid, tgt_pid = queue.popleft()
        if tgt_pid in visited: continue
        ref_di = mapping[ref_pid]
        tgt_pn = pattern.nodes[tgt_pid]

        # Step A (v4): 직접 → 간접 순으로 한 단계씩 넓히며 후보를 찾는다.
        # 가장 가까운 단계에서 임계값을 넘는 노드가 있으면 그 단계에서 고르고, 동점이면 위치가 가까운 노드.
        # (v3b는 3단계 후보를 한꺼번에 모아 집합 순서로 동점을 갈라, 직접 연결보다 먼 노드를 고를 수 있었음)
        adj = succ_map if d == "fwd" else pred_map
        best_i, best_s = None, 0.0
        frontier, seen = {ref_di}, {ref_di}
        for _ in range(3):
            level = set()
            for fi in frontier:
                for ni in adj.get(fi, []):
                    if ni not in seen: level.add(ni); seen.add(ni)
            scored = []
            for ci in level:
                if ci in rev: continue
                cn = idx_to_node.get(ci)
                if not cn: continue
                scored.append((_node_match_score(tgt_pn, cn, strict), -abs(ci - ref_di), ci))
            ok = [x for x in scored if x[0] >= threshold]
            if ok:
                sc, _, ci = max(ok)
                best_i, best_s = ci, sc
                break
            frontier = level
            if not frontier: break

        # Step B: adjacency 실패 시 토폴로지 근접성 폴백
        # (비rngd 중간 노드 때문에 SSA 연결이 끊긴 경우 대응)
        if best_i is None or best_s < threshold:
            proximity_range = 5  # 앞뒤 5개 노드 이내 탐색
            if d == "fwd":
                nearby = [n for n in nodes if ref_di < n.idx <= ref_di + proximity_range]
            else:
                nearby = [n for n in nodes if ref_di - proximity_range <= n.idx < ref_di]
            nearby.sort(key=lambda n: abs(n.idx - ref_di))   # 가까운 노드부터 (동점이면 가까운 쪽이 남음)
            for cn in nearby:
                if cn.idx in rev: continue
                s = _node_match_score(tgt_pn, cn, strict)
                if s > best_s: best_s = s; best_i = cn.idx

        if best_i is not None and best_s >= threshold:
            mapping[tgt_pid] = best_i; rev[best_i] = tgt_pid
            scores.append(best_s); visited.add(tgt_pid)
            for sp in tgt_pn.successors:
                if sp not in visited: queue.append(("fwd", tgt_pid, sp))
            for pp in tgt_pn.predecessors:
                if pp not in visited: queue.append(("bwd", tgt_pid, pp))

    ratio = len(mapping)/pattern.size if pattern.size>0 else 0.0
    avg = sum(scores)/len(scores) if scores else 0.0
    overall = ratio * avg
    if overall < threshold: return None
    return MatchResult(pattern_name=pattern.name, matched_indices=sorted(mapping.values()),
                       similarity=round(overall,4),
                       details={"match_ratio":round(ratio,4),"avg_node_similarity":round(avg,4),
                                "matched_count":len(mapping),"pattern_size":pattern.size,
                                "mapping":{str(k):v for k,v in mapping.items()}})


# =====================================================================
# 진입점 탐색
# =====================================================================

def _find_entry_anchors(pattern, nodes, idx_to_node, succ_map, pred_map, threshold, strict):
    entries = pattern.entry_nodes
    if not entries: return []
    if len(entries)==1:
        en = entries[0]
        return [{en.id:dn.idx} for dn in nodes if _node_match_score(en,dn,strict)>=threshold]

    # 다중 진입점: 같은 입력 공유 그룹
    anchors = []
    groups = {}
    for dn in nodes:
        for inp in dn.inputs:
            groups.setdefault(inp,[]).append(dn.idx)
    for grp in groups.values():
        if len(grp) < len(entries): continue
        a = {}; ok = True
        for en in entries:
            bi, bs = None, 0.0
            for di in grp:
                if di in a.values(): continue
                dn = idx_to_node.get(di)
                if not dn: continue
                s = _node_match_score(en, dn, strict)
                if s > bs: bs=s; bi=di
            if bi and bs >= threshold: a[en.id]=bi
            else: ok=False; break
        if ok and len(a)==len(entries): anchors.append(a)
    return anchors

def _find_subpattern_anchors(pattern, nodes, idx_to_node, succ_map, pred_map, sub_matches, threshold, strict):
    if not pattern.sub_patterns or not sub_matches: return []
    anchors = []
    for sub_name in pattern.sub_patterns:
        for sm in [m for m in sub_matches if m.pattern_name==sub_name]:
            sub_idx = sm.matched_indices
            if not sub_idx: continue
            anchor = {}
            for si in sub_idx:
                sdn = idx_to_node.get(si)
                if not sdn: continue
                for pn in pattern.nodes:
                    if pn.id in anchor: continue
                    s = _node_match_score(pn, sdn, strict)
                    if s >= threshold:
                        anchor[pn.id] = si; break
            if anchor: anchors.append(anchor)
    return anchors


# =====================================================================
# 패턴 탐색 (통합)
# =====================================================================

def find_patterns(pattern, nodes, threshold=0.7, strict=True, allow_overlap=False, sub_matches=None):
    idx_to_node = {n.idx:n for n in nodes}
    _, succ, pred = _build_adjacency(nodes)
    anchors = _find_entry_anchors(pattern, nodes, idx_to_node, succ, pred, threshold, strict)
    if sub_matches:
        anchors.extend(_find_subpattern_anchors(pattern, nodes, idx_to_node, succ, pred, sub_matches, threshold, strict))

    results = []; used = set()
    for a in anchors:
        if not allow_overlap and any(v in used for v in a.values()): continue
        m = _match_dag(pattern, nodes, a, succ, pred, idx_to_node, threshold, strict)
        if m:
            if not allow_overlap:
                if used & set(m.matched_indices): continue
                used.update(m.matched_indices)
            results.append(m)
    return results


# =====================================================================
# 컨텍스트 매칭 (RESIDUAL)
# =====================================================================

def find_context_patterns(pattern, nodes, pred_map=None, threshold=0.7):
    if pattern.size > 2: return []
    if not pred_map: _, _, pred_map = _build_adjacency(nodes)
    results = []
    for pn in pattern.nodes:
        for dn in nodes:
            if pn.op_type!="*" and pn.op_type!=dn.op_type: continue
            if pn.op_attr!="*" and pn.op_attr!=dn.op_attr: continue
            ctx = pn.context
            if not ctx: continue
            score = 1.0
            if "external_sources" in ctx:
                # v4: 한쪽 입력이 블록 바깥(함수 인자)이거나, 5노드 이상 앞에서 온 입력이면 Skip Connection으로 본다.
                preds = pred_map.get(dn.idx, [])
                gap = max((dn.idx - p for p in preds), default=0)
                if "arg" in (dn.input_sources or []): score *= 1.0
                elif gap >= 5: score *= 0.9
                else: score *= 0.3
            if score >= threshold:
                results.append(MatchResult(pattern_name=pattern.name, matched_indices=[dn.idx],
                                           similarity=round(score,4), details={"context_match":True}))
    return results


# =====================================================================
# 패턴 라이브러리
# =====================================================================

class PatternLibrary:
    def __init__(self): self.patterns = {}
    def add_pattern(self, p): self.patterns[p.name] = p
    def add_from_example(self, nodes, indices, name, desc="", sub_patterns=None):
        pg = extract_pattern_from_example(nodes, indices, name, desc)
        if sub_patterns: pg.sub_patterns = sub_patterns
        self.patterns[name] = pg; return pg

    def find_all(self, nodes, threshold=0.7, strict=True, allow_overlap=False):
        """세 단계로 찾는다: 체인 → DAG(하위 패턴 앵커) → 맥락.
        노드를 이미 쓴 매칭이 있으면 같은 종류(kind)끼리만 겹치지 않게 한다.
        종류가 다르면 겹칠 수 있다 (예: FFN 안의 SiLU·GELU)."""
        results = []; phase1 = []
        used = {}   # kind → 이미 쓴 노드
        _, _, pred_map = _build_adjacency(nodes)
        taken = lambda p, m: used.get(self.kind_of(p.name), set()) & set(m.matched_indices)
        # 같은 템플릿의 후보가 겹치면 노드가 촘촘하게 모인(범위가 좁은) 쪽을 먼저 채택
        # (직접 연결이 끊겨 보조 경로로 먼 노드까지 이어진 후보보다, 바로 이웃한 후보가 맞음)
        compact = lambda ms: sorted(ms, key=lambda m: (max(m.matched_indices) - min(m.matched_indices), -m.similarity))
        def take(p, m):
            used.setdefault(self.kind_of(p.name), set()).update(m.matched_indices)

        small, chain, dag = [], [], []
        for p in self.patterns.values():
            if p.match_mode == "context" or (p.size <= 2 and not p.is_chain): small.append(p)   # 맥락 패턴(Residual)은 맥락 단계에서만
            elif p.is_chain: chain.append(p)
            else: dag.append(p)

        # Phase 1: 체인
        for p in sorted(chain, key=lambda x:x.size, reverse=True):
            for m in compact(find_patterns(p, nodes, threshold, strict, allow_overlap)):
                if not allow_overlap and taken(p, m): continue
                results.append(m); phase1.append(m)
                if not allow_overlap: take(p, m)

        # Phase 2: DAG (하위 패턴 앵커, overlap 허용)
        for p in sorted(dag, key=lambda x:x.size, reverse=True):
            for m in compact(find_patterns(p, nodes, threshold, strict, True, phase1)):
                new = set(m.matched_indices) - used.get(self.kind_of(p.name), set())
                if len(new) >= max(1, m.details.get("pattern_size",1)*0.2):
                    results.append(m)
                    if not allow_overlap: take(p, m)

        # Phase 3: 컨텍스트
        for p in small:
            for m in find_context_patterns(p, nodes, pred_map, threshold):
                if not allow_overlap and taken(p, m): continue
                results.append(m)
                if not allow_overlap: take(p, m)
        return self._suppress_overlaps(results)

    def kind_of(self, name):
        p = self.patterns.get(name)
        return (p.kind or p.name) if p else name

    def _suppress_overlaps(self, results):
        """같은 종류(kind)의 매칭끼리 작은 쪽 노드의 절반 이상이 겹치면 큰 매칭만 남긴다.
        (예: GeGLU FFN 안에서 단순형 GELU FFN 템플릿이 일부를 다시 잡는 경우)"""
        kept = []
        for m in sorted(results, key=lambda r: (len(r.matched_indices), r.similarity), reverse=True):
            s = set(m.matched_indices); k = self.kind_of(m.pattern_name)
            if any(self.kind_of(o.pattern_name) == k and len(s & set(o.matched_indices)) >= len(s) / 2 for o in kept):
                continue
            kept.append(m)
        return sorted(kept, key=lambda r: min(r.matched_indices))

    def save(self, path):
        with open(path,"w") as f: json.dump({n:p.to_dict() for n,p in self.patterns.items()},f,indent=2,ensure_ascii=False)
    def load(self, path):
        with open(path) as f: data = json.load(f)
        for n,d in data.items(): self.patterns[n] = PatternGraph.from_dict(d)
    def summary(self):
        print(f"=== Pattern Library ({len(self.patterns)} patterns) ===")
        for name, pg in self.patterns.items():
            tags = ["chain" if pg.is_chain else "DAG"]
            if pg.size<=2 and not pg.is_chain: tags.append("ctx")
            if pg.sub_patterns: tags.append(f"sub={pg.sub_patterns}")
            ent = [n.op_attr or n.op_type for n in pg.entry_nodes]
            print(f"  {name}: {pg.size}n [{','.join(tags)}] entry=[{','.join(ent)}]")


# =====================================================================
# 비교 유틸리티
# =====================================================================

def compare_with_baseline(bl_groups, prop_matches, verbose=True):
    bl={}
    for g in bl_groups:
        nm = g.name if hasattr(g,'name') else g.get('name','')
        ix = set(g.node_indices if hasattr(g,'node_indices') else g.get('node_indices',[]))
        bl.setdefault(nm,[]).append(ix)
    pr={}
    for m in prop_matches:
        pr.setdefault(m.pattern_name,[]).append(set(m.matched_indices))

    names = sorted(set(list(bl.keys())+list(pr.keys())))
    s = {"baseline_total":sum(len(v) for v in bl.values()),
         "proposed_total":sum(len(v) for v in pr.values()),
         "exact_matches":0,"partial_matches":0,"missed":0,"new_detections":0}
    det = {}
    for nm in names:
        blist, plist = bl.get(nm,[]), pr.get(nm,[])
        d = {"baseline_count":len(blist),"proposed_count":len(plist),"exact_matches":0,"partial_matches":0}
        for bi in blist:
            best = max((len(bi&pi)/len(bi) if bi else 0 for pi in plist),default=0)
            if best>=0.8: d["exact_matches"]+=1; s["exact_matches"]+=1
            elif best>0.3: d["partial_matches"]+=1; s["partial_matches"]+=1
            else: s["missed"]+=1
        if not blist and plist: s["new_detections"]+=len(plist)
        det[nm]=d

    if verbose:
        print("=== Baseline vs Proposed ===")
        for k in ["baseline_total","proposed_total","exact_matches","partial_matches","missed","new_detections"]:
            print(f"  {k}: {s[k]}")
        print()
        for nm,d in det.items():
            ic = "✅" if d["exact_matches"]==d["baseline_count"] else ("⚠️" if d["exact_matches"]+d["partial_matches"]>0 else "❌")
            print(f"  {ic} {nm}: bl={d['baseline_count']}, pr={d['proposed_count']}, ex={d['exact_matches']}, pt={d['partial_matches']}")
    return {"summary":s,"patterns":det}

def matches_to_fusion_groups(ms):
    return [{"name":m.pattern_name,"node_indices":m.matched_indices,"similarity":m.similarity} for m in ms]

if __name__=="__main__":
    print("graph_pattern_matcher.py v3b — isolated node recovery + refined indirect connections")
