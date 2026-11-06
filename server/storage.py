"""
실행 이력 저장소 — 변환 결과와 커널 소스를 SQLite에 보관합니다.

테이블 구조는 운영 서버의 pipeline_results.db와 동일하게 유지하며,
최근 100건만 남기고 오래된 기록은 트리거로 자동 정리합니다.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime

from rngd_pipeline import config

MAX_RUNS = 100
LIST_LIMIT = 20

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS conversion_runs (
    run_id TEXT PRIMARY KEY,
    model_name TEXT,
    model_type TEXT,
    timestamp TEXT,
    coverage_pct REAL,
    converted_count INTEGER,
    total_count INTEGER,
    pass_l12 INTEGER,
    total_l12 INTEGER,
    pass_l3 INTEGER,
    total_l3 INTEGER,
    fusion_json TEXT,
    markers_json TEXT,
    rngd_ir_structure TEXT,
    linalg_ir_structure TEXT,
    status TEXT,
    model_sub_type TEXT,
    model_code_hash TEXT,
    model_code TEXT,
    model_arch TEXT,
    model_params TEXT,
    verify_scale TEXT,
    core_ops TEXT
);
CREATE TABLE IF NOT EXISTS kernel_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    module_name TEXT,
    source_code TEXT,
    max_diff REAL,
    passed INTEGER,
    FOREIGN KEY (run_id) REFERENCES conversion_runs(run_id)
);
CREATE TRIGGER IF NOT EXISTS cleanup_old_runs
AFTER INSERT ON conversion_runs
BEGIN
    DELETE FROM kernel_sources WHERE run_id IN (
        SELECT run_id FROM conversion_runs
        ORDER BY timestamp DESC
        LIMIT -1 OFFSET {MAX_RUNS}
    );
    DELETE FROM conversion_runs WHERE run_id NOT IN (
        SELECT run_id FROM conversion_runs
        ORDER BY timestamp DESC
        LIMIT {MAX_RUNS}
    );
END;
"""

_RUN_COLUMNS = (
    "run_id", "model_name", "model_type", "timestamp", "coverage_pct",
    "converted_count", "total_count", "pass_l12", "total_l12", "pass_l3", "total_l3",
    "fusion_json", "markers_json", "rngd_ir_structure", "linalg_ir_structure",
    "status", "model_sub_type", "model_code_hash", "model_code",
    "model_arch", "model_params", "verify_scale", "core_ops",
)


def _connect() -> sqlite3.Connection:
    config.VAR_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.RESULTS_DB)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with closing(_connect()) as con:
        con.executescript(_SCHEMA)
        con.commit()


def save_run(run_data: dict) -> str | None:
    """변환 결과 1건을 저장하고 run_id를 반환합니다. 실패해도 예외를 던지지 않습니다."""
    run_id = run_data.get("run_id") or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    row = {
        "run_id": run_id,
        "model_name": run_data.get("model_name", ""),
        "model_type": run_data.get("model_type", ""),
        "timestamp": datetime.now().isoformat(),
        "coverage_pct": run_data.get("coverage_pct", 0),
        "converted_count": run_data.get("converted_count", 0),
        "total_count": run_data.get("total_count", 0),
        "pass_l12": run_data.get("pass_l12", 0),
        "total_l12": run_data.get("total_l12", 0),
        "pass_l3": run_data.get("pass_l3", 0),
        "total_l3": run_data.get("total_l3", 0),
        "fusion_json": json.dumps(run_data.get("fusion", {}), ensure_ascii=False),
        "markers_json": json.dumps(run_data.get("markers", {}), ensure_ascii=False),
        "rngd_ir_structure": run_data.get("rngd_ir", ""),
        "linalg_ir_structure": run_data.get("linalg_ir", ""),
        "status": run_data.get("status", "success"),
        "model_sub_type": run_data.get("model_sub_type", ""),
        "model_code_hash": run_data.get("model_code_hash", ""),
        "model_code": run_data.get("model_code", ""),
        "model_arch": run_data.get("model_arch", ""),
        "model_params": run_data.get("model_params", ""),
        "verify_scale": run_data.get("verify_scale", ""),
        "core_ops": run_data.get("core_ops", ""),
    }
    try:
        init_db()
        with closing(_connect()) as con:
            placeholders = ",".join("?" * len(_RUN_COLUMNS))
            con.execute(
                f"INSERT OR REPLACE INTO conversion_runs ({','.join(_RUN_COLUMNS)}) VALUES ({placeholders})",
                tuple(row[c] for c in _RUN_COLUMNS),
            )
            # 같은 run_id로 다시 저장할 때 커널 기록이 중복되지 않도록 먼저 비움
            con.execute("DELETE FROM kernel_sources WHERE run_id = ?", (run_id,))
            con.executemany(
                "INSERT INTO kernel_sources (run_id, module_name, source_code, max_diff, passed) "
                "VALUES (?,?,?,?,?)",
                [(run_id, k.get("module_name", ""), k.get("source_code", ""),
                  k.get("max_diff", 0), 1 if k.get("passed", False) else 0)
                 for k in run_data.get("kernels", [])],
            )
            con.commit()
        return run_id
    except Exception as e:  # noqa: BLE001 — 이력 저장 실패가 검증 결과를 막지 않도록
        print(f"[실행 이력 저장 실패] {e}")
        return None


def list_runs(limit: int = LIST_LIMIT) -> list[dict]:
    init_db()
    with closing(_connect()) as con:
        rows = con.execute(
            "SELECT run_id, model_name, model_type, timestamp, coverage_pct, converted_count, "
            "total_count, pass_l12, total_l12, pass_l3, total_l3, status "
            "FROM conversion_runs ORDER BY timestamp DESC LIMIT ?", (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    init_db()
    with closing(_connect()) as con:
        row = con.execute("SELECT * FROM conversion_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        for key in ("fusion_json", "markers_json"):
            if result.get(key):
                try:
                    result[key] = json.loads(result[key])
                except (TypeError, ValueError):
                    pass
        kernels = con.execute(
            "SELECT module_name, max_diff, passed, length(source_code) AS source_size "
            "FROM kernel_sources WHERE run_id = ?", (run_id,),
        ).fetchall()
    result["kernels"] = [dict(k) for k in kernels]
    return result


def get_run_text(run_id: str, field: str) -> str | None:
    """IR 원문 조회 — field는 'rngd' 또는 'linalg'"""
    column = {"rngd": "rngd_ir_structure", "linalg": "linalg_ir_structure"}[field]
    init_db()
    with closing(_connect()) as con:
        row = con.execute(f"SELECT {column} FROM conversion_runs WHERE run_id = ?", (run_id,)).fetchone()
    return row[0] if row and row[0] else None


def get_kernel_source(run_id: str, module: str) -> str | None:
    init_db()
    with closing(_connect()) as con:
        row = con.execute(
            "SELECT source_code FROM kernel_sources WHERE run_id = ? AND module_name = ?",
            (run_id, module),
        ).fetchone()
    return row[0] if row and row[0] else None
