from dataclasses import dataclass
from typing import Optional

class VideoAnalysisState:
    """
    Persistent, cross-frame state for the video analysis loop.
    This object owns all temporal decisions and user-driven intent.
    """
    def __init__(self):
        self.paused = True
        self.selecting = True
        self.primary_idx = None
        self.primary_center = None
        self.primary_bbox = None
        self.frame_idx = 0