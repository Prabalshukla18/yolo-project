"""
detector.py — YOLOv8 object detection engine for FORENIX AI.

Uses the Ultralytics library to run YOLOv8 inference on video frames
and returns structured detection records ready for database insertion.

Usage
-----
    detector = YOLODetector(model="yolov8x.pt", confidence=0.5)
    for meta, frame in extractor:
        records = detector.detect(meta, frame, video_id=1, camera_id="CAM-01")
        db.insert_detection_batch(conn, records)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics not installed. Run: pip install ultralytics"
    )


# ─────────────────────────────────────────────
# Detection result dataclass
# ─────────────────────────────────────────────

@dataclass
class Detection:
    object: str
    confidence: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int

    # ── derived helpers ───────────────────────
    @property
    def area(self) -> int:
        return self.bbox_w * self.bbox_h

    @property
    def center(self) -> tuple[int, int]:
        return (self.bbox_x + self.bbox_w // 2,
                self.bbox_y + self.bbox_h // 2)


# ─────────────────────────────────────────────
# Flag heuristics — tweak per deployment
# ─────────────────────────────────────────────

# Objects that should trigger a forensic flag when confidence is high
FLAGGED_CLASSES = {"person", "backpack", "handbag", "suitcase", "knife", "gun"}
FLAG_CONFIDENCE_THRESHOLD = 0.88


def _should_flag(obj: str, confidence: float) -> bool:
    return obj in FLAGGED_CLASSES and confidence >= FLAG_CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────

class YOLODetector:
    """
    Wraps a YOLOv8 model and provides a clean detect() interface.

    Parameters
    ----------
    model           : Model weight path or name ('yolov8n.pt', 'yolov8x.pt', …).
    confidence      : Minimum confidence to keep a detection (0–1).
    iou             : IoU threshold for NMS.
    device          : 'cpu', 'cuda', 'mps', or '' (auto).
    classes         : Optional list of COCO class IDs to restrict detection to.
                      e.g. [0, 2, 7] → person, car, truck only.
    imgsz           : Inference image size (pixels, square).
    half            : Use FP16 inference on CUDA for ~2× speed.
    """

    # COCO class name → friendly label remapping
    _LABEL_MAP: dict[str, str] = {
        "person": "person",
        "bicycle": "bicycle",
        "car": "car",
        "motorcycle": "motorcycle",
        "bus": "bus",
        "truck": "truck",
        "backpack": "bag",
        "handbag": "bag",
        "suitcase": "bag",
        "knife": "knife",
    }

    def __init__(
        self,
        model: str = "yolov8n.pt",
        confidence: float = 0.50,
        iou: float = 0.45,
        device: str = "",
        classes: Optional[list[int]] = None,
        imgsz: int = 640,
        half: bool = False,
    ):
        print(f"[detector] Loading model: {model} …")
        self.model = YOLO(model)
        self.confidence = confidence
        self.iou = iou
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.classes = classes
        self.imgsz = imgsz
        self.half = half and self.device == "cuda"
        print(f"[detector] Ready  device={self.device}  conf≥{confidence}")

    # ── public ────────────────────────────────

    def detect(
        self,
        frame_meta,          # FrameMeta from extractor
        frame: np.ndarray,   # BGR numpy array
        *,
        video_id: int,
        camera_id: str = "CAM-01",
    ) -> list[dict]:
        """
        Run inference on a single frame.

        Returns a list of dicts compatible with db.insert_detection_batch().
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )

        rows: list[dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_label = result.names.get(cls_id, "unknown")
                label = self._LABEL_MAP.get(raw_label, raw_label)
                conf = float(box.conf[0])

                # xyxy → xywh
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1

                rows.append({
                    "video_id":   video_id,
                    "camera_id":  camera_id,
                    "frame_num":  frame_meta.frame_num,
                    "timestamp_s": frame_meta.timestamp_s,
                    "wall_time":  frame_meta.wall_time,
                    "object":     label,
                    "confidence": round(conf, 4),
                    "bbox_x":     bx,
                    "bbox_y":     by,
                    "bbox_w":     bw,
                    "bbox_h":     bh,
                    "flagged":    int(_should_flag(label, conf)),
                })

        return rows

    def detect_batch(
        self,
        frames: list[tuple],   # list of (FrameMeta, np.ndarray)
        *,
        video_id: int,
        camera_id: str = "CAM-01",
    ) -> list[dict]:
        """Process multiple frames in a single model call (faster on GPU)."""
        all_rows: list[dict] = []
        for meta, frame in frames:
            all_rows.extend(self.detect(meta, frame,
                                        video_id=video_id,
                                        camera_id=camera_id))
        return all_rows

    # ── private ───────────────────────────────

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
