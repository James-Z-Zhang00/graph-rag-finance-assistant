"""
Source-content router — direct Neo4j access.
"""

from typing import Dict
from fastapi import APIRouter, HTTPException

from db.connection import get_db_manager
from services.kg_service import get_source_content, get_source_file_info
from services.neo4j_batch import BatchProcessor
from models.schemas import (
    SourceRequest, SourceResponse,
    SourceInfoBatchRequest, ContentBatchRequest,
)

router = APIRouter()


@router.post("/source", response_model=SourceResponse)
async def source(request: SourceRequest):
    content = get_source_content(request.source_id)
    return SourceResponse(content=content)


@router.post("/source_info")
async def source_info(request: SourceRequest):
    return get_source_file_info(request.source_id)


@router.post("/content_batch", response_model=Dict)
async def get_content_batch(request: ContentBatchRequest):
    try:
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        return BatchProcessor.get_content_batch(request.chunk_ids, driver)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get content batch: {str(e)}")


@router.post("/source_info_batch", response_model=Dict)
async def get_source_info_batch(request: SourceInfoBatchRequest):
    try:
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        return BatchProcessor.get_source_info_batch(request.source_ids, driver)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get source info batch: {str(e)}")
