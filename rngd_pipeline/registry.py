"""
registry.py
===========
rngd_registry.db를 읽어, 대시보드(dashboard_app.py)와 파이프라인(e2e_pipeline.py)이
필요로 하는 데이터 구조를 반환하는 헬퍼 모듈.

설계 원칙:
- 이 모듈만 DB를 알고 있다. dashboard_app.py / e2e_pipeline.py는 이 모듈을 import하기만 한다.
- 반환 타입은 기존 dict 구조와 호환을 유지해 대시보드 코드 수정을 최소화한다.
- 모든 함수는 읽기 전용. DB는 registry/rngd_registry.sql 로부터 최초 1회 생성된다.
"""

import sqlite3
from pathlib import Path
from functools import lru_cache

from . import config

DB_PATH = config.REGISTRY_DB


def build_db(force: bool = False) -> Path:
    """SQL 덤프(registry/rngd_registry.sql)로부터 레지스트리 DB를 생성한다."""
    if DB_PATH.exists() and not force:
        return DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(config.REGISTRY_SQL.read_text(encoding="utf-8"))
        con.commit()
    finally:
        con.close()
    tmp.replace(DB_PATH)
    return DB_PATH


def _con() -> sqlite3.Connection:
    build_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


# ──────────────────────────────────────────────────────────────────────────────
# 파이프라인(e2e_pipeline.py)이 사용하는 함수
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_op_mappings() -> dict:
    """
    rewrite_to_rngd()가 참조하는 linalg → rngd 매핑을 DB에서 읽어 반환한다.

    반환 예시:
    {
        "arith_binary": {"arith.addf": "add", "arith.subf": "sub", ...},
        "math_unary":   {"math.rsqrt": "rsqrt"},
        "contraction":  {"linalg.batch_matmul": "batch_gemm", ...},
        "special":      {"pow2": {"family": "elementwise", "source_op": "linalg.generic(math.powf, 지수=2.0)"}},
    }
    """
    con = _con()
    rows = con.execute(
        "SELECT op_name, family, source_op, target_op FROM ops WHERE status='done'"
    ).fetchall()
    con.close()

    arith_binary, math_unary, contraction, special = {}, {}, {}, {}
    for r in rows:
        src = r["source_op"]
        op = r["op_name"]
        if r["family"] == "elementwise":
            if "arith." in src:
                # "linalg.generic(arith.addf)" → "arith.addf"
                inner = src.split("(")[-1].rstrip(")")
                arith_binary[inner] = op
            elif "math.powf" in src:
                special["pow2"] = {"family": r["family"], "source_op": src}
            elif "math." in src:
                inner = src.split("(")[-1].rstrip(")")
                math_unary[inner] = op
        elif r["family"] == "contraction":
            # "linalg.batch_matmul" → "batch_gemm"
            math_unary_key = src.split("(")[0].strip()
            contraction[math_unary_key] = op

    return {
        "arith_binary": arith_binary,
        "math_unary": math_unary,
        "contraction": contraction,
        "special": special,
    }


@lru_cache(maxsize=1)
def load_tolerances() -> dict[str, str]:
    """op_name → tolerance 문자열 매핑. 검증 로직에서 사용."""
    con = _con()
    rows = con.execute("SELECT op_name, tolerance FROM ops").fetchall()
    con.close()
    return {r["op_name"]: r["tolerance"] for r in rows}


# ──────────────────────────────────────────────────────────────────────────────
# 대시보드(dashboard_app.py)가 사용하는 함수
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_supported_ops() -> list[str]:
    """SUPPORTED_OPS 리스트를 DB에서 읽어 반환 (family, op_name 순 정렬)."""
    con = _con()
    rows = con.execute(
        "SELECT op_name FROM ops WHERE status='done' ORDER BY family, op_name"
    ).fetchall()
    con.close()
    # 대시보드 버튼 표시 순서를 맞추기 위해 family 내에서 원래 순서를 보존
    order = ["add", "sub", "mul", "div", "rsqrt", "pow2",
             "batch_gemm", "gemm", "dot_product"]
    db_ops = {r["op_name"] for r in rows}
    return [op for op in order if op in db_ops] + \
        [r["op_name"] for r in rows if r["op_name"] not in order]


@lru_cache(maxsize=1)
def load_op_groups() -> dict[str, str]:
    """OP_GROUPS 딕셔너리를 DB에서 읽어 반환."""
    con = _con()
    rows = con.execute("SELECT op_name, family FROM ops WHERE status='done'").fetchall()
    con.close()
    return {r["op_name"]: r["family"] for r in rows}


@lru_cache(maxsize=1)
def load_op_metadata() -> dict[str, dict]:
    """
    OP_METADATA 딕셔너리를 DB에서 재구성해 반환.
    기존 dashboard_app.py의 딕셔너리 구조와 완전히 호환된다.
    """
    con = _con()
    ops = con.execute("SELECT * FROM ops WHERE status='done'").fetchall()
    con.close()

    result = {}
    for row in ops:
        op = row["op_name"]
        family_label = _family_display(row["family"], op)
        meta: dict = {
            "family":       family_label,
            "source_op":    row["source_op"] + " → " + row["target_op"],
            "dtype":        row["dtype"],
            "tolerance":    row["tolerance"],
            "compares":     row["compares"] or "",
            "meaning_pass": row["meaning_pass"] or "",
            "meaning_fail": row["meaning_fail"] or "",
            "detail":       row["detail"] or "",
        }
        if row["experimental"]:
            meta["experimental"] = True
        result[op] = meta

    return result


def invalidate_cache() -> None:
    """DB가 변경된 뒤 캐시를 무효화할 때 사용."""
    load_op_mappings.cache_clear()
    load_tolerances.cache_clear()
    load_supported_ops.cache_clear()
    load_op_groups.cache_clear()
    load_op_metadata.cache_clear()


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _family_display(family: str, op_name: str) -> str:
    """DB의 'elementwise'/'contraction' 값을 대시보드 표시용 한국어로 변환."""
    unary_ops = {"rsqrt", "pow2"}
    if family == "elementwise":
        if op_name in unary_ops:
            return "Elementwise (Family A, 단항)"
        return "Elementwise (Family A)"
    if family == "contraction":
        labels = {
            "gemm":        "Contraction (Family B, 배치 없음)",
            "dot_product": "Contraction (Family B, 벡터 내적)",
        }
        return labels.get(op_name, "Contraction (Family B)")
    return family


# ──────────────────────────────────────────────────────────────────────────────
# CLI 디버그 출력
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== supported_ops ===")
    print(load_supported_ops())

    print("\n=== op_groups ===")
    print(load_op_groups())

    print("\n=== op_mappings (파이프라인용) ===")
    m = load_op_mappings()
    for k, v in m.items():
        print(f"  {k}: {v}")
