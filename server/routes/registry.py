"""변환 규칙 레지스트리 조회 API (읽기 전용)"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from rngd_pipeline import registry

router = APIRouter(prefix="/api/registry", tags=["registry"])

_SEARCH_FIELDS = ("op_name", "source_op", "target_op", "detail", "compares")


@router.get("")
def list_rules(q: Optional[str] = Query(None, description="연산 이름, 표준 IR·NPU 특화 IR 연산, 커널 체인에서 검색"),
               family: Optional[str] = Query(None, description="elementwise | contraction"),
               dtype: Optional[str] = Query(None, description="f32 | bf16")):
    """변환 규칙 목록. 예: /api/registry?family=elementwise&q=rsqrt"""
    rules = registry.list_rules()
    if family:
        rules = [r for r in rules if r["family"] == family]
    if dtype:
        rules = [r for r in rules if r["dtype"] == dtype]
    if q:
        key = q.lower()
        rules = [r for r in rules if any(key in (r[f] or "").lower() for f in _SEARCH_FIELDS)]
    return {"count": len(rules), "rules": rules}


@router.get("/{op_name}")
def get_rule(op_name: str):
    """연산 하나의 변환 규칙"""
    rule = registry.get_rule(op_name)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"레지스트리에 '{op_name}' 연산이 없습니다.")
    return rule
