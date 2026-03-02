import os
import cv2
import numpy as np

from app.detection.detector import Detection


def get_frame_quadrant(x1: float, y1: float, x2: float, y2: float,
                       frame_width: int, frame_height: int) -> str:
    """
    Return which quadrant the center of a bounding box falls in.
    Used for spatial querying: "show me cars in the top-right".
    """
    cx = (x1 + x2) / 2 / frame_width
    cy = (y1 + y2) / 2 / frame_height

    if cx < 0.33:
        h_zone = "left"
    elif cx < 0.66:
        h_zone = "center"
    else:
        h_zone = "right"

    if cy < 0.4:
        v_zone = "top"
    elif cy < 0.7:
        v_zone = "mid"
    else:
        v_zone = "bottom"

    if h_zone == "center" and v_zone == "mid":
        return "center"
    return f"{v_zone}-{h_zone}"


def extract_crop(
    frame: np.ndarray,
    detection: Detection,
    padding_fraction: float = 0.1,
) -> np.ndarray:
    """
    Extract and return a cropped image of a detected object with padding.

    padding_fraction=0.1 adds 10% of the bounding box dimensions as padding
    on each side, giving the attribute extractor more context around the object.
    """
    h, w = frame.shape[:2]

    # Add padding
    bw = detection.x2 - detection.x1
    bh = detection.y2 - detection.y1
    pad_x = bw * padding_fraction
    pad_y = bh * padding_fraction

    x1 = max(0, int(detection.x1 - pad_x))
    y1 = max(0, int(detection.y1 - pad_y))
    x2 = min(w, int(detection.x2 + pad_x))
    y2 = min(h, int(detection.y2 + pad_y))

    # Guard against degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return frame[
            max(0, int(detection.y1)):min(h, int(detection.y2)),
            max(0, int(detection.x1)):min(w, int(detection.x2)),
        ]

    return frame[y1:y2, x1:x2]


def save_crop(
    crop: np.ndarray,
    base_data_dir: str,
    video_filename: str,
    frame_second: float,
    track_id: int,
    object_class: str,
) -> str:
    """
    Save a crop to disk and return its path.

    Directory structure:
      data/keyframes/{video_stem}/crops/{object_class}_{track_id}_{second}.jpg

    Crops are small JPEGs — quality 90 keeps detail for attribute extraction.
    """
    video_stem = os.path.splitext(video_filename)[0]
    crops_dir = os.path.join(base_data_dir, "keyframes", video_stem, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    filename = f"{object_class}_{track_id}_{frame_second:.2f}.jpg"
    path = os.path.join(crops_dir, filename)

    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def normalize_bbox(x1: float, y1: float, x2: float, y2: float,
                   frame_width: int, frame_height: int) -> tuple[float, float, float, float]:
    """Convert pixel bbox to normalized 0-1 coordinates."""
    return (
        x1 / frame_width,
        y1 / frame_height,
        x2 / frame_width,
        y2 / frame_height,
    )