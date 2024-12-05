from fastapi import APIRouter, HTTPException
from loguru import logger

from src import dependency
from src.core.models.rag_request import Query
from src.core.models.rag_response import ResponseModel

router = APIRouter(tags=["rag"])


@router.post("/_v1/rag", tags=["rag_v1"])
async def test_rag_v1(q: Query):
    try:
        answer = dependency.rag_pipeline.generate_response(q.query, top_k=q.top_k)
        return ResponseModel(query=q.query, response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
