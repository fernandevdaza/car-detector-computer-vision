from ultralytics import YOLO
from typing import Optional
class State():
    yolo_model: Optional[YOLO]


state = State()