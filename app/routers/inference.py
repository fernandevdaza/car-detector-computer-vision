import mimetypes
import os
from fastapi import APIRouter, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from starlette import status
from langchain.messages import HumanMessage
import base64
from app.core.agent import agent
from app.controllers.infer_with_image_controller import infer_with_image_controller
from app.controllers.infer_with_video_controller import infer_with_video_controller
from typing import Optional, List, Literal
import tempfile
from app.models.DetectResponse import DetectResponse
from starlette.concurrency import run_in_threadpool

router = APIRouter(
    prefix="/inference",
    tags=["inferences"]
)

@router.post("/car-with-image")
async def infer_car_with_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Debe ser una imagen")
    result = await infer_with_image_controller(file)
    
    return {"message": result.get("messages")[1].content}

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
            infer_with_video_controller,
            temp_path,
            frame_stride,
            conf,
            iou,
            max_crops,
            min_crop_side,
            thumb_width,
            vehicle_types
        )
        return JSONResponse(result.model_dump())

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


