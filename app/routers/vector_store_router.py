from fastapi import APIRouter, HTTPException, Depends
from app.services.data_processor import get_pdf_processor, PDFProcessor
from app.models.request_models import (
    VectorStoreRequest,
    VectorStoreResponse,
)
from app.services.vector_store_service import get_vector_store_service, VectorStoreService
import asyncio

router = APIRouter(prefix="/load_vector_store", tags=["vector_store"])


def get_processor() -> PDFProcessor:
    return get_pdf_processor()

async def get_vector_store() -> VectorStoreService:
    return await get_vector_store_service()

@router.get("/initialize_load_vector_store", response_model=VectorStoreResponse)
async def load_vector_store(
    vector_store: VectorStoreService = Depends(get_vector_store),
    processor: PDFProcessor = Depends(get_processor),
):
    try:
        documents = processor.load_persisted_chunks()
        ingested = await vector_store.add_documents(documents)
        return VectorStoreResponse(len_documents=ingested)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
