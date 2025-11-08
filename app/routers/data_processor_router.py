from fastapi import APIRouter, HTTPException, Depends
from app.services.data_processor import get_pdf_processor, PDFProcessor
from app.models.request_models import (
    PDFProcessRequest,
    PDFProcessResponse
)

router = APIRouter(prefix="/data_processor", tags=["pdfs"])


def get_processor() -> PDFProcessor:
    return get_pdf_processor()

@router.post("/process", response_model=PDFProcessResponse)
async def process_all_pdfs(req: PDFProcessRequest, processor: PDFProcessor = Depends(get_processor)):
    try:
        result = processor.process_all_pdfs(
            split=req.split,
            glob_pattern=req.glob_pattern,
            parallel=req.parallel,
            max_workers=req.max_workers,
            persist=req.persist,
            persist_dir=req.persist_dir,
            persist_format=req.persist_format,
        )
        return PDFProcessResponse(
            total_files=result.get("total_files", 0),
            total_pages=result.get("total_pages", 0),
            total_chunks=result.get("total_chunks", 0),
            processed_files=result.get("processed_files", []),
            processed_at=result.get("processed_at"),
            persisted=result.get("persisted"),
            persist_dir=result.get("persist_dir"),
            persist_error=result.get("persist_error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
