from pydantic import BaseModel
from typing import List

class Crop(BaseModel):
    id: str
    label: str
    score: float
    frame_idx: int
    bbox_xywh: List[int]
    thumb_b64: str