from fastapi import APIRouter, HTTPException, Depends
from app.services.hybrid_fusion_service import get_hybrid_fusion_service, HybridFusionService
from app.models.request_models import FusionRequest, FusionResponse

router = APIRouter(prefix="/hybrid_fusion", tags=["hybrid_fusion"])

async def get_service() -> HybridFusionService:
    return await get_hybrid_fusion_service()

@router.post("/fuse", response_model=FusionResponse)
async def fuse(req: FusionRequest, service: HybridFusionService = Depends(get_service)):
    try:
        result = await service.fuse(req)
        return FusionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
