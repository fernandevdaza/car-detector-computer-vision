from pydantic import BaseModel
from app.models.Crop import Crop
from typing import List

class DetectResponse(BaseModel):
    total_frames: int
    total_crops: int
    crops: List[Crop]