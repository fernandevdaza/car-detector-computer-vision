from ultralytics import YOLO
from dotenv import load_dotenv
import numpy as np
import os


load_dotenv()

yolo_model = None

def init_yolo_model():
    
    global yolo_model
    try: 
        model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        if not yolo_model:
            yolo_model = YOLO(model_name)

        warm_up_model(yolo_model)

        return yolo_model
    except Exception as e:
        raise Exception(f"Hubo un problema cargando el modelo {model_name}: {e}")
    
def warm_up_model(model: YOLO):
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)

    

