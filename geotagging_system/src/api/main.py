import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.exceptions import GeotaggingError, MetadataParseError, CoordinateTransformError, ReportGenerationError
from src.api.endpoints import router

logger = logging.getLogger(__name__)

# Application state
APP_START_TIME = None
TASK_REGISTRY = {}  # task_id -> task state dict
SURVEY_DATA = {}    # survey_id -> parsed ping data + results

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown handler."""
    global APP_START_TIME
    APP_START_TIME = time.time()
    logger.info("Geotagging API starting up...")
    yield
    logger.info("Geotagging API shutting down...")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Marine Debris Geotagging API",
        description="REST API for geotagging underwater marine debris detections from side-scan sonar imagery.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers for custom exceptions
    @app.exception_handler(GeotaggingError)
    async def geotagging_error_handler(request: Request, exc: GeotaggingError):
        return JSONResponse(
            status_code=500,
            content={"error": "GeotaggingError", "detail": str(exc), "status_code": 500}
        )
    
    @app.exception_handler(MetadataParseError)
    async def parse_error_handler(request: Request, exc: MetadataParseError):
        return JSONResponse(
            status_code=400,
            content={"error": "MetadataParseError", "detail": str(exc), "status_code": 400}
        )
        
    @app.exception_handler(CoordinateTransformError)
    async def transform_error_handler(request: Request, exc: CoordinateTransformError):
        return JSONResponse(
            status_code=500,
            content={"error": "CoordinateTransformError", "detail": str(exc), "status_code": 500}
        )
        
    @app.exception_handler(ReportGenerationError)
    async def report_error_handler(request: Request, exc: ReportGenerationError):
        return JSONResponse(
            status_code=500,
            content={"error": "ReportGenerationError", "detail": str(exc), "status_code": 500}
        )
    
    # Include router
    app.include_router(router)
    
    return app

app = create_app()
