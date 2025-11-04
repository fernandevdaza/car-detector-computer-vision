from typing import List, Optional
from app.models.DetectResponse import DetectResponse
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from app.utils.video_processing import _to_data_url, _to_thumbnail
from app.core.state import state
from app.models.Crop import Crop
from fastapi import HTTPException
import cv2

COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

def infer_with_video_controller(
    video_path: str,
    frame_stride: int,
    conf: float,
    iou: float,
    max_crops: int,
    min_crop_side: int,
    thumb_width: int,
    vehicle_types: Optional[List[str]]
) -> DetectResponse:
    
    model: YOLO = state.yolo_model
    wanted_cls_ids = { i for i, name in COCO_VEHICLE_CLASSES.items() if name in set(vehicle_types or [])}

    cap = cv2.VideoCapture(video_path)    
    if not cap.isOpened():
        raise HTTPException(400, detail="No se pudo abrir el video")
    
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    crops: List[Crop] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        if not results:
            frame_idx += 1
            continue

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]


        for b_idx, box in enumerate(r.boxes):
            cls_id = int(box.cls.item())
            if cls_id not in wanted_cls_ids:
                continue

            score = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < min_crop_side or bh < min_crop_side:
                continue
            
            crop_img = frame[y1:y2, x1:x2].copy()

            thumb = _to_thumbnail(crop_img, thumb_width)
            data_url = _to_data_url(thumb)

            crops.append(
                Crop(
                    id=f"f{frame_idx}_b{b_idx}",
                    label=COCO_VEHICLE_CLASSES.get(cls_id, str(cls_id)),
                    score=round(score, 4),
                    frame_idx=frame_idx,
                    bbox_xywh=[x1, y1, bw, bh],
                    thumb_b64=data_url
                )
            )
            if len(crops) >= max_crops:
                break
        if len(crops) >= max_crops:
            break
        frame_idx += 1

    cap.release()

    return DetectResponse(total_frames=total_frames, total_crops=len(crops), crops=crops)    
            
                

