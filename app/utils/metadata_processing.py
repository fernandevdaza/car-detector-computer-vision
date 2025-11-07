# deps:
# poetry add pillow-heif piexif
# y tener ffprobe instalado en el sistema (ffmpeg)

import io, os, re, json, tempfile, subprocess
from typing import Tuple
from fastapi import APIRouter, UploadFile, File, HTTPException

from PIL import Image, UnidentifiedImageError
import piexif
from pillow_heif import register_heif_opener, read_heif

register_heif_opener()  # habilita PIL para abrir .heic/.heif

router = APIRouter()

# ---------- Utilidades comunes ----------
def _ensure_exif_header(raw: bytes) -> bytes:
    return raw if raw.startswith(b"Exif\x00\x00") else (b"Exif\x00\x00" + raw)

def _dms_to_deg(dms, ref: bytes) -> float:
    to_float = lambda x: x[0] / x[1]
    deg = to_float(dms[0]) + to_float(dms[1]) / 60 + to_float(dms[2]) / 3600
    if ref in (b'S', b'W'): deg = -deg
    return float(deg)

def _parse_iso6709(s: str) -> Tuple[float | None, float | None]:
    # acepta +lat-lon[/alt], también con decimales
    m = re.search(r'([+-]\d{1,3}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)(?:[+-]\d{1,3}(?:\.\d+)?/?)?', s)
    if not m: return None, None
    return float(m.group(1)), float(m.group(2))

def _parse_lat_lon_pairs(s: str) -> Tuple[float | None, float | None]:
    # intenta "lat,lon" o "lat lon"
    m = re.search(r'([+-]?\d{1,3}(?:\.\d+)?)[,\s]+([+-]?\d{1,3}(?:\.\d+)?)', s)
    if not m: return None, None
    return float(m.group(1)), float(m.group(2))

# ---------- Imágenes (JPEG/HEIC/HEIF) ----------
def process_image_metadata_from_bytes(data: bytes, filename: str | None, content_type: str | None):
    if not data:
        return {"has_gps": False, "note": "Archivo vacío"}

    ct = (content_type or "").lower()
    name = (filename or "").lower()
    is_heif = ct in {"image/heic", "image/heif"} or name.endswith((".heic", ".heif"))
    is_jpeg = ct == "image/jpeg" or name.endswith((".jpg", ".jpeg"))

    # Validación
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except UnidentifiedImageError:
        return {"has_gps": False, "note": "Bytes no corresponden a una imagen válida"}
    except Exception as e:
        return {"has_gps": False, "note": f"No se pudo validar imagen: {e}"}

    # JPEG → EXIF directo
    if is_jpeg:
        try:
            exif = piexif.load(data)
            gps = exif.get("GPS") or {}
            lat = _dms_to_deg(gps[piexif.GPSIFD.GPSLatitude],  gps[piexif.GPSIFD.GPSLatitudeRef])
            lon = _dms_to_deg(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef])
            return {"has_gps": True, "lat": lat, "lon": lon, "source": "EXIF-JPEG"}
        except Exception:
            pass

    # HEIF/HEIC → intenta img.info (EXIF/XMP)
    try:
        img2 = Image.open(io.BytesIO(data))  # reabrir tras verify()
        info = getattr(img2, "info", {}) or {}
        exif_bytes = info.get("exif")
        xmp_bytes  = info.get("xmp") or info.get("XML:com.adobe.xmp")
        if exif_bytes:
            try:
                exif = piexif.load(_ensure_exif_header(exif_bytes))
                gps = exif.get("GPS") or {}
                lat = _dms_to_deg(gps[piexif.GPSIFD.GPSLatitude],  gps[piexif.GPSIFD.GPSLatitudeRef])
                lon = _dms_to_deg(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef])
                return {"has_gps": True, "lat": lat, "lon": lon, "source": "EXIF-img.info"}
            except Exception:
                pass
        if xmp_bytes:
            txt = xmp_bytes.decode("utf-8", errors="ignore")
            lat, lon = _parse_iso6709(txt)
            if lat is None:
                lat, lon = _parse_lat_lon_pairs(txt)
            if lat is not None and lon is not None:
                return {"has_gps": True, "lat": lat, "lon": lon, "source": "XMP-img.info"}
    except Exception:
        pass

    # Fallback: read_heif + metadatos (si tu versión expone algo)
    try:
        hf = read_heif(data)
        meta = getattr(hf, "metadata", None)
        if isinstance(meta, list) and meta:
            exif_bytes = next((m["data"] for m in meta if (m.get("type") or "").lower()=="exif" and m.get("data")), None)
            xmp_bytes  = next((m["data"] for m in meta if (m.get("type") or "").lower()=="xmp"  and m.get("data")), None)
            if exif_bytes:
                try:
                    exif = piexif.load(_ensure_exif_header(exif_bytes))
                    gps = exif.get("GPS") or {}
                    lat = _dms_to_deg(gps[piexif.GPSIFD.GPSLatitude],  gps[piexif.GPSIFD.GPSLatitudeRef])
                    lon = _dms_to_deg(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef])
                    return {"has_gps": True, "lat": lat, "lon": lon, "source": "EXIF-read_heif"}
                except Exception:
                    pass
            if xmp_bytes:
                txt = xmp_bytes.decode("utf-8", errors="ignore")
                lat, lon = _parse_iso6709(txt)
                if lat is None:
                    lat, lon = _parse_lat_lon_pairs(txt)
                if lat is not None and lon is not None:
                    return {"has_gps": True, "lat": lat, "lon": lon, "source": "XMP-read_heif"}
    except Exception:
        pass

    return {"has_gps": False, "note": "Sin GPS en EXIF/XMP"}

# ---------- Videos (MP4/MOV) ----------
def process_video_metadata_from_path(path: str):
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-print_format", "json",
                "-show_entries", "format_tags:stream_tags",
                path
            ],
            stderr=subprocess.STDOUT,
        )
        meta = json.loads(out.decode("utf-8", "ignore"))
    except FileNotFoundError:
        return {"has_gps": False, "note": "ffprobe no está instalado"}
    except subprocess.CalledProcessError as e:
        return {"has_gps": False, "note": f"ffprobe error: {e.output.decode('utf-8','ignore')[:200]}..."}

    # aplanar tags (format + streams)
    def find_tags(d):
        tags = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if k.lower() == "tags" and isinstance(v, dict):
                    tags.update(v)
                elif isinstance(v, dict):
                    tags.update(find_tags(v))
                elif isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            tags.update(find_tags(it))
        return tags

    tags = {k.lower(): str(v) for k, v in find_tags(meta).items()}

    # candidatos más comunes
    candidates = [
        "com.apple.quicktime.location.iso6709",
        "location",
        "com.android.location",
        "com.apple.quicktime.gpscoordinates",
        "gpslatitude",
        "gpslongitude",
    ]

    # 1) ISO6709 (+lat-lon/)
    for key in candidates:
        if key in tags:
            lat, lon = _parse_iso6709(tags[key])
            if lat is not None:
                return {"has_gps": True, "lat": lat, "lon": lon, "source": f"video:{key}-iso6709"}

    # 2) pares "lat,lon" en cualquier tag
    for key, val in tags.items():
        lat, lon = _parse_lat_lon_pairs(val)
        if lat is not None:
            return {"has_gps": True, "lat": lat, "lon": lon, "source": f"video:{key}-pair"}

    # 3) separados
    if "gpslatitude" in tags and "gpslongitude" in tags:
        try:
            return {"has_gps": True, "lat": float(tags["gpslatitude"]), "lon": float(tags["gpslongitude"]), "source": "video:separate-gps"}
        except Exception:
            pass

    return {"has_gps": False, "note": "Sin GPS en tags del contenedor"}