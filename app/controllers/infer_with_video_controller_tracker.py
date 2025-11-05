from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
from fastapi import HTTPException
import numpy as np
from ultralytics import YOLO
from app.utils.video_processing import _to_thumbnail, _to_data_url
from app.models.Crop import Crop
from app.models.DetectResponse import DetectResponse
from app.core.state import state

def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = max(ax2, bx2), max(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2- inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(a_area + b_area - inter + 1e-6)

def var_laplacian(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def center_penalty(bbox: Tuple[int, int, int, int], w: int, h: int) -> float:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx, ny = cx / w, cy / h
    dx, dy = abs(nx - 0.5), abs(ny - 0.5)
    dist = (dx*dx + dy*dy) ** 0.5
    return float(max(0.0, 1.0 - dist * 2.0))

def quality_score(conf: float, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2-x1), max(1, y2-y1)
    area_rel = (bw * bh) / float(w * h)
    sharp = var_laplacian(frame[y1:y2, x1:x2]) if (bw > 5 and bh > 5) else 0.0

    conf_n = float(conf)
    area_n = float(min(1.0, area_rel / 0.05))
    sharp_n = float(min(1.0, sharp / 200.0))
    center_n = center_penalty(bbox, w, h)

    return 0.4 * conf_n + 0.25 * area_n + 0.25 * sharp_n + 0.10 * center_n

def phash(img: np.ndarray, size: int = 32, dct_size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(gray.astype(np.float32))
    dct_low = dct[:dct_size, :dct_size]
    med = float(np.median(dct_low))
    bits = (dct_low > med).astype(np.uint64)
    h = 0
    for b in bits.flatten():
        h = (h << 1) | int(b)
    return int(h)

def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))

def hsv_hist(img: np.ndarray, bins = (16, 16, 16)) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten()
    return hist.astype(np.float32)

def chi2(a: np.ndarray, b: np.ndarray, eps = 1e-10) -> float:
    d = (a - b) ** 2 / (a + b + eps)
    return float(0.5 * np.sum(d))

@dataclass
class Track:
    id: int
    cls_id: int
    last_bbox: Tuple[int, int, int, int]
    last_frame: int

class IoUTracker:
    def __init__(self, iou_match: float = 0.3, max_age: int = 20):
        self.iou_match = iou_match
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, 
               detections: List[Tuple[int, int, int, int]], 
               frame_idx: int) -> List[Tuple[int, int, int, int]]:
        outputs: List[Tuple[int, int, int, int]] = []
        used = set()

        for tid, tr in list(self.tracks.items()):
            best_j, best_iou = -1, 0.0
            for j, (c,b) in enumerate(detections):
                if j in used or c != tr.cls_id:
                    continue
                iou = iou_xyxy(tr.last_bbox, b)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= self.iou_match:
                c, b = detections[best_j]
                used.add(best_j)
                tr.last_bbox, tr.last_frame = b, frame_idx
                outputs.append((tid, c, b))
            else:
                if frame_idx - tr.last_frame > self.max_age:
                    del self.tracks[tid]
            
        for j, (c, b) in enumerate(detections):
            if j in used:
                continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = Track(tid, c, b, frame_idx)
            outputs.append((tid, c, b))
        return outputs
    
@dataclass
class CropOut:
    track_id: int
    cls_id: int
    label: str
    score: float
    frame_idx: int
    bbox_xyxy: Tuple[int, int, int, int]
    crop_img: np.ndarray
    quality: float

COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def infer_with_video_controller_tracker(
    video_path: str,
    frame_stride: int,
    conf: float,
    iou: float,
    max_crops: int,
    min_crop_side: int,
    thumb_width: int,
    vehicle_types: Optional[List[str]],
):
    model: YOLO = state.yolo_model
    wanted = { i for i, n in COCO_VEHICLE_CLASSES.items() if n in set(vehicle_types or [])}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, detail="No se pudo abrir el video")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)


    tracker = IoUTracker(iou_match=0.3, max_age=15)
    best_by_track: Dict[int, CropOut] = {}


    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue


        res = model.predict(frame, conf=conf, iou=iou, verbose=False)
        dets: List[Tuple[int,Tuple[int,int,int,int]]] = []
        if res and res[0].boxes is not None:
            r = res[0]
            h, w = frame.shape[:2]
            for box in r.boxes:
                cls_id = int(box.cls.item())
                if cls_id not in wanted:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(w-1,x2), min(h-1,y2)
                bw,bh = x2-x1, y2-y1
                if bw < min_crop_side or bh < min_crop_side:
                    continue
                dets.append((cls_id, (x1,y1,x2,y2)))
        # tracking
        assigns = tracker.update(dets, frame_idx)


        # scoring + guardar mejor por track
        for tid, cls_id, bbox in assigns:
            x1,y1,x2,y2 = bbox
            crop = frame[y1:y2, x1:x2].copy()
            q = quality_score(conf=1.0, bbox=bbox, frame=frame) # usa 1.0 aquí; si quieres, mezcla con box.conf (se puede guardar aparte)
        # etiqueta
            label = COCO_VEHICLE_CLASSES.get(cls_id, str(cls_id))
            cand = CropOut(tid, cls_id, label, score=1.0, frame_idx=frame_idx, bbox_xyxy=bbox, crop_img=crop, quality=q)
            prev = best_by_track.get(tid)
            if prev is None or cand.quality > prev.quality:
                best_by_track[tid] = cand


        if len(best_by_track) >= max_crops: # límite duro temprano
            pass


        frame_idx += 1


    cap.release()
    # --- deduplicación global por pHash + hist ---
    reps = list(best_by_track.values())
# ordena por calidad descendente
    reps.sort(key=lambda c: c.quality, reverse=True)


    kept: List[CropOut] = []
    phashes: List[int] = []
    hists: List[np.ndarray] = []


    PHASH_HAMMING_TH = 10
    HIST_CHI2_TH = 0.25


    for c in reps:
        ph = phash(c.crop_img)
        hist = hsv_hist(c.crop_img)
        similar = False
        for i,k in enumerate(kept):
            if hamming(ph, phashes[i]) < PHASH_HAMMING_TH:
            # muy parecido perceptualmente; verifica también histograma
                if chi2(hist, hists[i]) < HIST_CHI2_TH:            
                    similar = True
                    break
        if not similar:
            kept.append(c)
            phashes.append(ph)
            hists.append(hist)
        if len(kept) >= max_crops:
            break


# --- construir respuesta ---
    crops_out = []
    for c in kept:
        x1,y1,x2,y2 = c.bbox_xyxy
        bw,bh = x2-x1, y2-y1
        thumb = _to_thumbnail(c.crop_img, thumb_width)
        data_url = _to_data_url(thumb)
        crops_out.append(Crop(
        id=f"t{c.track_id}_f{c.frame_idx}",
        label=c.label,
        score=round(float(c.quality), 4),
        frame_idx=int(c.frame_idx),
        bbox_xywh=[int(x1), int(y1), int(bw), int(bh)],
        thumb_b64=data_url,
        ))


    return DetectResponse(
    total_frames=int(total_frames),
    total_crops=len(crops_out),
    crops=crops_out,
    )