"""
extractor.py — Frame extraction from CCTV video using OpenCV.

Supports:
  • Fixed interval sampling  (every N frames)
  • Time-based sampling      (every N seconds)
  • Keyframe-only mode       (scene-change detection via frame difference)

Usage
-----
    extractor = FrameExtractor("footage/cam01.mp4", sample_every_n=5)
    for meta, frame in extractor:
        # meta: FrameMeta  |  frame: np.ndarray (BGR)
        process(meta, frame)
"""

import cv2
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Generator, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class VideoMeta:
    filename: str
    path: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_s: float


@dataclass
class FrameMeta:
    frame_num: int
    timestamp_s: float
    wall_time: str          # ISO-8601 string


# ─────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────

class FrameExtractor:
    """
    Iterates over a video file and yields (FrameMeta, np.ndarray) tuples.

    Parameters
    ----------
    video_path      : Path to the input video file.
    sample_every_n  : Yield every Nth frame (default 1 = every frame).
    sample_every_s  : Yield one frame every N seconds (overrides sample_every_n).
    keyframes_only  : Use scene-change detection; ignore sample_every_n.
    diff_threshold  : Mean absolute pixel diff to count as a scene change (0–255).
    start_time      : Treat this datetime as the video's wall-clock start time.
                      Defaults to now if not provided.
    resize_to       : Optional (width, height) to resize frames before yielding.
    """

    def __init__(
        self,
        video_path: str,
        sample_every_n: int = 1,
        sample_every_s: Optional[float] = None,
        keyframes_only: bool = False,
        diff_threshold: float = 25.0,
        start_time: Optional[datetime] = None,
        resize_to: Optional[Tuple[int, int]] = None,
    ):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.keyframes_only = keyframes_only
        self.diff_threshold = diff_threshold
        self.resize_to = resize_to
        self.start_time = start_time or datetime.utcnow()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.video_meta = VideoMeta(
            filename=os.path.basename(video_path),
            path=os.path.abspath(video_path),
            fps=fps,
            width=w,
            height=h,
            total_frames=tc,
            duration_s=tc / fps if fps else 0.0,
        )

        # Resolve effective sample interval in frames
        if sample_every_s is not None:
            self._interval = max(1, int(round(fps * sample_every_s)))
        else:
            self._interval = max(1, sample_every_n)

    # ── public ────────────────────────────────

    def __iter__(self) -> Generator[Tuple[FrameMeta, np.ndarray], None, None]:
        cap = cv2.VideoCapture(self.video_path)
        fps = self.video_meta.fps
        prev_gray: Optional[np.ndarray] = None
        frame_num = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_num += 1
                ts_s = (frame_num - 1) / fps if fps else 0.0
                wall = (self.start_time + timedelta(seconds=ts_s)).isoformat()

                should_yield = self._should_yield(
                    frame_num, frame, prev_gray
                )

                if should_yield:
                    if self.resize_to:
                        frame = cv2.resize(frame, self.resize_to,
                                           interpolation=cv2.INTER_AREA)
                    yield FrameMeta(frame_num, ts_s, wall), frame

                # Update previous frame for scene-diff
                if self.keyframes_only:
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        finally:
            cap.release()

    def frames_count(self) -> int:
        """Estimated number of frames that will be yielded."""
        if self.keyframes_only:
            return -1   # can't predict scene changes ahead of time
        return max(1, self.video_meta.total_frames // self._interval)

    # ── private ───────────────────────────────

    def _should_yield(self, frame_num: int,
                      frame: np.ndarray,
                      prev_gray: Optional[np.ndarray]) -> bool:
        if self.keyframes_only:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                return True
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            return diff >= self.diff_threshold
        return (frame_num % self._interval) == 0

    # ── convenience ───────────────────────────

    def save_frames(self, output_dir: str) -> int:
        """Extract and save all sampled frames as JPEG files."""
        os.makedirs(output_dir, exist_ok=True)
        saved = 0
        for meta, frame in self:
            path = os.path.join(output_dir,
                                f"frame_{meta.frame_num:07d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        return saved


# ─────────────────────────────────────────────
# Quick CLI usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <video_path> [sample_every_n]")
        sys.exit(1)

    path = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    ex = FrameExtractor(path, sample_every_n=every)
    print(f"Video : {ex.video_meta.filename}")
    print(f"FPS   : {ex.video_meta.fps:.2f}")
    print(f"Size  : {ex.video_meta.width}×{ex.video_meta.height}")
    print(f"Frames: {ex.video_meta.total_frames}  (~{ex.frames_count()} sampled)")

    for meta, frame in ex:
        print(f"  frame={meta.frame_num:>7}  ts={meta.timestamp_s:.3f}s  "
              f"wall={meta.wall_time}", end="\r")
    print("\nDone.")
