import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logger import SingletonLogger
from app.core.logging_middleware import LoggingMiddleware
from app.routers import health, data_processor_router, vector_store_router, knowledge_graph_router, hybrid_fusion_router

logger = SingletonLogger().get_logger()

# Create FastAPI application instance
logger.info("Starting FastAPI application...")
app = FastAPI(root_path=settings.APP_ROOT_PATH)

# Add CORS middleware
logger.info("Setting up CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# Include routers
logger.info("Including routers...")
app.include_router(health.router)
app.include_router(data_processor_router.router)
app.include_router(vector_store_router.router)
app.include_router(knowledge_graph_router.router)
app.include_router(hybrid_fusion_router.router)

# Root endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Graph + Vector RAG API",
        "version": "1.0.0",
        "status": "active"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
