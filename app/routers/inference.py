import mimetypes
import os
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from requests import Session
from app.controllers.infer_with_image_controller import infer_with_image_controller
from app.controllers.infer_with_video_controller_tracker import infer_with_video_controller_tracker
from typing import Annotated, Optional, List, Literal
import tempfile
from app.models.db.models import Cars
from app.models.DetectResponse import DetectResponse
from starlette.concurrency import run_in_threadpool
from app.utils.metadata_processing import process_image_metadata_from_bytes, process_video_metadata_from_path
from app.core.db import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]

router = APIRouter(
    prefix="/inference",
    tags=["inferences"]
)

@router.post("/car-with-image")
async def infer_car_with_image(
    db: db_dependency,
    file: UploadFile = File(...),
    lat: Annotated[Optional[float], Form()] = None,
    lon: Annotated[Optional[float], Form()] = None,
    source_video_id: Annotated[Optional[str], Form()] = None,
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Debe ser una imagen")

    data = await file.read()

    # tu inferencia (si acepta bytes)
    result = await infer_with_image_controller(data)

    # metadatos desde el crop Y/O heredados del video
    meta = process_image_metadata_from_bytes(data, filename=file.filename, content_type=file.content_type)

    # Si el front envió lat/lon del video, úsalos como fallback o como “source of truth”
    if lat is not None and lon is not None:
        meta = {**meta, "video_lat": lat, "video_lon": lon}

    if source_video_id:
        meta["source_video_id"] = source_video_id

    car_model = Cars(
        brand=result.brand,
        model_name=result.model_name,
        year=result.year,
        lat=lat if lat else meta.get("lat"),
        lng=lon if lon else meta.get("lon")
    )

    db.add(car_model)
    db.commit()

    return {"message": result, "metadata": meta}


@router.post("/car-with-video")
async def infer_car_with_video(
    file: UploadFile = File(..., description="Video file (e.g., .mp4, .mov)"),
    frame_stride: int = Query(5, ge=1, le=60, description="Process 1 of every N frames"),
    conf: float = Query(0.35, ge=0.05, le=0.95, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.1, le=0.9, description="IoU for NMS"),
    max_crops: int = Query(50, ge=1, le=500, description="Hard cap on returned crops"),
    min_crop_side: int = Query(48, ge=16, le=2048, description="Skip crops smaller than this (pixels)"),
    thumb_width: int = Query(256, ge=64, le=1024, description="Thumbnail width for base64 previews"),
    vehicle_types: Optional[List[Literal["car", "motorcycle", "bus", "truck"]]] = Query(
        ["car", "truck"], description="which vehicle classes to keep"
        ),
):
    ct = (file.content_type or "").strip().lower()
    name = (file.filename or "").strip().lower()

    looks_like_video = (
        ct.startswith("video/") or
        name.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")) or
        (mimetypes.guess_type(name)[0] or "").startswith("video/")
    )
    if not looks_like_video:
        raise HTTPException(400, detail="Debe enviar un video (content-type 'video/*' o extensión válida)")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".mp4") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
    except Exception as e:
        raise HTTPException(500, detail=f"No se puede guardar el video: {e}")
    
    try:
        result: DetectResponse = await run_in_threadpool(
            infer_with_video_controller_tracker,
            temp_path,
            frame_stride,
            conf,
            iou,
            max_crops,
            min_crop_side,
            thumb_width,
            vehicle_types
        )
        metadata_result = process_video_metadata_from_path(temp_path)
        payload = result.model_dump()
        payload["metadata"] = metadata_result
        return JSONResponse(payload)

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


