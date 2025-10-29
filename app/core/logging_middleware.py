from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time

from app.core.logger import SingletonLogger

logger = SingletonLogger().get_logger()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        logger.info(f"Incoming request: {request.method} {request.url}")

        try:
            response = await call_next(request)
        except Exception as e:
            logger.exception(f"Error while handling request: {e}")
            raise

        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"Completed {request.method} {request.url.path} "
            f"with status {response.status_code} in {process_time:.2f} ms"
        )

        return response
