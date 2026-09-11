import uuid
import os
import tempfile
import time
import json
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from src.api.schemas import (
    UploadSonarRequest, UploadResponse, ProcessDetectionsRequest,
    ProcessDetectionsResponse, TaskStatusResponse, ReportResponse,
    HealthResponse, ErrorResponse
)
# Using direct import since we are in the same module level and need to access the dictionaries initialized in main
# Avoid circular import by accessing the global module state dicts
import src.api.main as api_main

from src.models import MissionMetadata, Detection, SonarImageInfo, SurveyReport
from src.pipeline.geotagging_engine import GeotaggingEngine
from src.reporting.report_generator import ReportGenerator
from src.exceptions import GeotaggingError

router = APIRouter()
logger = logging.getLogger(__name__)


def background_parse_file(task_id: str, survey_id: str, file_path: str):
    """Background task to parse sonar file."""
    try:
        api_main.TASK_REGISTRY[task_id]["status"] = "processing"
        api_main.TASK_REGISTRY[task_id]["progress"] = 0.5
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)
        
        # Simulated parsing
        time.sleep(2)
        
        api_main.SURVEY_DATA[survey_id] = {"parsed_file": file_path, "status": "ready"}
        api_main.TASK_REGISTRY[task_id]["status"] = "completed"
        api_main.TASK_REGISTRY[task_id]["progress"] = 1.0
        api_main.TASK_REGISTRY[task_id]["result"] = {"message": "File parsed successfully"}
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        api_main.TASK_REGISTRY[task_id]["status"] = "failed"
        api_main.TASK_REGISTRY[task_id]["error"] = str(e)
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)


def background_process_detections(task_id: str, survey_id: str, request: ProcessDetectionsRequest):
    """Background task to process detections."""
    try:
        api_main.TASK_REGISTRY[task_id]["status"] = "processing"
        api_main.TASK_REGISTRY[task_id]["progress"] = 0.1
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)
        
        # Simulate processing logic with GeotaggingEngine
        # engine = GeotaggingEngine()
        time.sleep(2)
        
        api_main.TASK_REGISTRY[task_id]["status"] = "completed"
        api_main.TASK_REGISTRY[task_id]["progress"] = 1.0
        api_main.TASK_REGISTRY[task_id]["result"] = {"message": f"Processed {len(request.detections)} detections."}
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)
        
        if survey_id not in api_main.SURVEY_DATA:
            api_main.SURVEY_DATA[survey_id] = {}
        # Serialize detections for storage
        api_main.SURVEY_DATA[survey_id]["detections"] = [d.model_dump() for d in request.detections]
        
    except Exception as e:
        logger.error(f"Error processing detections: {e}")
        api_main.TASK_REGISTRY[task_id]["status"] = "failed"
        api_main.TASK_REGISTRY[task_id]["error"] = str(e)
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)


def background_batch_process(task_id: str, survey_id: str, file_paths: List[str]):
    """Background task to process a batch of files."""
    try:
        api_main.TASK_REGISTRY[task_id]["status"] = "processing"
        api_main.TASK_REGISTRY[task_id]["progress"] = 0.5
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)
        
        time.sleep(2)
        
        api_main.TASK_REGISTRY[task_id]["status"] = "completed"
        api_main.TASK_REGISTRY[task_id]["progress"] = 1.0
        api_main.TASK_REGISTRY[task_id]["result"] = {"message": f"Batch processed {len(file_paths)} files."}
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        api_main.TASK_REGISTRY[task_id]["status"] = "failed"
        api_main.TASK_REGISTRY[task_id]["error"] = str(e)
        api_main.TASK_REGISTRY[task_id]["updated_at"] = datetime.now(timezone.utc)


@router.post(
    "/upload_sonar_data",
    response_model=UploadResponse,
    status_code=202,
    tags=["Sonar Data"],
    summary="Upload sonar file and metadata",
    description="Upload a sonar data file (XTF, CSV, JSON) with metadata for processing."
)
async def upload_sonar_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Sonar data file"),
    survey_id: str = Form(..., description="Unique survey identifier"),
    sonar_system: str = Form(default="klein_3000"),
    vessel: str = Form(default="Unknown"),
):
    """
    Upload sonar file. Saves to temp location and queues background parsing.
    Returns task_id for progress tracking.
    Stream-writes file chunks to disk to handle large files.
    """
    task_id = str(uuid.uuid4())
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{survey_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    api_main.TASK_REGISTRY[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
        "updated_at": None,
    }
    
    background_tasks.add_task(background_parse_file, task_id, survey_id, file_path)

    return UploadResponse(
        task_id=task_id,
        survey_id=survey_id,
        filename=file.filename or "unknown",
        status="queued",
        message="File uploaded and queued for processing."
    )

@router.post(
    "/process_detections",
    response_model=ProcessDetectionsResponse,
    status_code=202,
    tags=["Detection Processing"],
    summary="Submit AI detections for geotagging"
)
async def process_detections(
    request: ProcessDetectionsRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit detection results from AI model for geotagging.
    Requires sonar data to have been uploaded first (survey_id must exist).
    Queues background geotagging processing.
    """
    if request.survey_id not in api_main.SURVEY_DATA:
        raise HTTPException(status_code=404, detail="Survey ID not found. Please upload sonar data first.")
        
    task_id = str(uuid.uuid4())
    api_main.TASK_REGISTRY[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
        "updated_at": None,
    }
    
    background_tasks.add_task(background_process_detections, task_id, request.survey_id, request)

    return ProcessDetectionsResponse(
        task_id=task_id,
        survey_id=request.survey_id,
        status="processing",
        num_detections=len(request.detections),
        message="Detections queued for geotagging."
    )

@router.get(
    "/get_report/{survey_id}",
    tags=["Reports"],
    summary="Fetch survey report"
)
async def get_report(
    survey_id: str,
    format: str = Query(default="json", description="Report format", enum=["json", "csv", "geojson", "gpkg"])
):
    """
    Retrieve generated report for a survey.
    Returns JSON inline or file download for binary formats.
    """
    if survey_id not in api_main.SURVEY_DATA:
        raise HTTPException(status_code=404, detail="Survey ID not found.")
        
    try:
        if format == "json":
            return {"survey_id": survey_id, "format": format, "data": api_main.SURVEY_DATA.get(survey_id)}
        else:
            temp_dir = tempfile.gettempdir()
            fake_report_path = os.path.join(temp_dir, f"report_{survey_id}.{format}")
            with open(fake_report_path, "w") as f:
                f.write(f"Fake {format} report content for {survey_id}")
            return FileResponse(path=fake_report_path, filename=f"report_{survey_id}.{format}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/get_geojson/{survey_id}",
    tags=["Reports"],
    summary="Fetch GeoJSON layer"
)
async def get_geojson(survey_id: str):
    """Return GeoJSON FeatureCollection for a survey's detections."""
    if survey_id not in api_main.SURVEY_DATA:
        raise HTTPException(status_code=404, detail="Survey ID not found.")
        
    # Placeholder for actual geojson extraction logic
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    return JSONResponse(content=geojson_data)

@router.post(
    "/batch_process",
    status_code=202,
    tags=["Batch Processing"],
    summary="Process multiple sonar files"
)
async def batch_process(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    survey_id: str = Form(...),
    sonar_system: str = Form(default="klein_3000"),
):
    """Upload and process multiple sonar files as a batch."""
    task_id = str(uuid.uuid4())
    temp_dir = tempfile.gettempdir()
    saved_files = []
    
    try:
        for file in files:
            file_path = os.path.join(temp_dir, f"batch_{survey_id}_{file.filename}")
            with open(file_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    f.write(chunk)
            saved_files.append(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save files: {str(e)}")
        
    api_main.TASK_REGISTRY[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
        "updated_at": None,
    }
    
    background_tasks.add_task(background_batch_process, task_id, survey_id, saved_files)
    
    return {
        "task_id": task_id,
        "survey_id": survey_id,
        "status": "queued",
        "message": f"{len(files)} files uploaded and queued for batch processing."
    }

@router.get(
    "/task_status/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Tasks"],
    summary="Check task status"
)
async def task_status(task_id: str):
    """Check the status of an async processing task."""
    if task_id not in api_main.TASK_REGISTRY:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_data = api_main.TASK_REGISTRY[task_id]
    return TaskStatusResponse(**task_data)

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check"
)
async def health_check():
    """System health check endpoint."""
    uptime = time.time() - (api_main.APP_START_TIME or time.time())
    active_tasks = sum(1 for t in api_main.TASK_REGISTRY.values() if t.get("status") in ["queued", "processing"])
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        active_tasks=active_tasks
    )
