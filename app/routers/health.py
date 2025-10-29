from fastapi import APIRouter

router = APIRouter(prefix="", tags=["health"])

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
