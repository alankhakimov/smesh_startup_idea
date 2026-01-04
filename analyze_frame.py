import cv2
import numpy as np
from ultralytics import YOLO

def analyze_frame(frame, model, conf_threshold=0.4, dim_background=True):
    """
    Runs fighter detection on a frame and returns a frame with detected objects

    Parameters
    ----------
    frame : np.ndarray
        BGR image from OpenCV
    conf_threshold : float
        YOLO confidence threshold
    dim_background : bool
        Whether to dim non-fighter regions

    Returns
    -------
    output_frame : np.ndarray
        Original frame with fighter highlight overlaid
    """

    results = model(frame, conf=conf_threshold, verbose=False)[0]

    # Filter for persons (class 0)
    persons = [
        box for box in results.boxes
        if int(box.cls) == 0
    ]

    if not persons:
        return frame.copy()

    output = frame.copy()

    if dim_background:
        # Dim entire frame first
        dimmed = (frame * 0.3).astype(np.uint8)
        output = dimmed.copy()

    for box in persons:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if dim_background:
            # Restore each person region
            output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return output

