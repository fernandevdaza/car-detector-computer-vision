import cv2
import numpy as np
import base64

def _to_thumbnail(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / float(w)
    new_w = target_w
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _to_data_url(img: np.ndarray) -> str:
    # JPEG with reasonable quality; you can tweak if artifacts appear
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("No se pudo codificar JPG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
