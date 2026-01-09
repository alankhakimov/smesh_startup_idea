import cv2
import numpy as np

#main pipeline
def draw_bbox(output, box, color=(0, 255, 0), thickness=2):
    """Draw bounding box on the frame."""
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
    return output, (x1, y1, x2, y2)

def draw_tracked_bbox(frame, bbox, color=(255, 0, 255), thickness=2):
    """
    Draw a bounding box on the frame.

    Args:
        frame (np.ndarray): Image to draw on.
        bbox (tuple): (x1, y1, x2, y2).
        color (tuple): BGR color (default: magenta).
        thickness (int): Line thickness.
    """
    if bbox is None:
        return frame

    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    return frame

def draw_keypoints(output, xy_array, conf_array, conf_thresh=0.9, color=(0, 0, 255), radius=3):
    """Draw keypoints as circles."""
    for i in range(xy_array.shape[0]):
        x, y = xy_array[i]
        conf = conf_array[i]
        if conf > conf_thresh:
            cv2.circle(output, (int(x), int(y)), radius, color, -1)
    return output

def draw_skeleton(output, xy_array, conf_array, skeleton, conf_thresh=0.9, color=(255, 0, 0), thickness=2):
    """Draw structural skeleton lines between keypoints."""
    for start_idx, end_idx in skeleton:
        if conf_array[start_idx] > conf_thresh and conf_array[end_idx] > conf_thresh:
            x0, y0 = xy_array[start_idx]
            x1, y1 = xy_array[end_idx]
            cv2.line(output, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
    return output


def render_detections(frame, detections, draw_skeleton_flag=True,
                      conf_thresh=0.3, fighter_idx=None):
    """
    Draw bounding boxes, keypoints, and skeletons.
    Highlights the fighter at fighter_idx with a yellow skeleton.

    Args:
        frame (np.array): The image frame.
        detections (list of dict): Each dict has "bbox" and "keypoints".
        draw_skeleton_flag (bool): Whether to draw skeletons.
        conf_thresh (float): Confidence threshold for keypoints.
        fighter_idx (int or None): Index of detection to highlight in yellow.
    """
    output = frame.copy()

    # Structural skeleton (COCO indices)
    skeleton_structural = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        kpts = det["keypoints"]

        # Draw bounding box (always green)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if draw_skeleton_flag:
            xy_array = kpts.xy.cpu().numpy().reshape(-1, 2)
            conf_array = kpts.conf.cpu().numpy().flatten()

            # Determine skeleton color
            if fighter_idx is not None and i == fighter_idx:
                skeleton_color = (0, 255, 255)  # yellow
            else:
                skeleton_color = (255, 0, 0)  # blue

            output = draw_keypoints(
                output, xy_array, conf_array, conf_thresh=conf_thresh
            )
            output = draw_skeleton(
                output, xy_array, conf_array,
                skeleton_structural, conf_thresh=conf_thresh,
                color=skeleton_color
            )

    return output

def get_primary_detection_idx(detections, state, alpha=0.001):
    """
    Determine the primary fighter index for the current frame.
    Uses both center distance and optional bbox area similarity from state.

    Args:
        detections (list of dict): Each dict has "bbox" and "keypoints".
        state (VideoAnalysisState): Contains primary_center and primary_bbox.
        alpha (float): Weight for area difference.

    Returns:
        int: index of the primary fighter. None if no detections or primary_center not set.
    """
    # Ensure we have a reference primary
    if state.primary_center is None or len(detections) == 0:
        return None

    primary_center_np = np.array(state.primary_center, dtype=float)

    # Compute reference area from stored primary_bbox
    if state.primary_bbox is not None:
        x1, y1, x2, y2 = state.primary_bbox
        primary_area = (x2 - x1) * (y2 - y1)
    else:
        primary_area = None

    scores = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bbox_center = np.array([(x1 + x2)/2, (y1 + y2)/2], dtype=float)
        center_dist = np.linalg.norm(bbox_center - primary_center_np)

        # Area similarity penalty
        if primary_area is not None:
            area = (x2 - x1) * (y2 - y1)
            area_diff = abs(area - primary_area)
            score = center_dist + alpha * area_diff
        else:
            score = center_dist

        scores.append(score)

    # Return the index of the detection with minimum score
    return int(np.argmin(scores))

def extract_detections(frame, model, conf_threshold=0.3):
    """
    Run YOLO pose model and return structured detections only.
    No drawing, no visualization.
    """
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    detections = []

    if results.keypoints is None:
        return detections

    for box, kpts in zip(results.boxes, results.keypoints):
        if int(box.cls) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "keypoints": kpts
        })

    return detections

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["last_click"] = (x, y)

def select_detection_from_click(detections,click_xy):
    """
    Return the index of the detection whose bounding box contains the click.
    If no bounding box contains the click, return None.
    """
    cx, cy = click_xy
    candidates = []

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]

        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, idx))

    if not candidates:
        return None

    # smallest bounding box wins
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def reassign_primary(detections, state, mouse_state):
    """
    Waits for the user to click on the new primary fighter and updates the state.
    """
    print("Click on the new primary fighter...")
    state.selecting = True

    while state.selecting:
        cv2.waitKey(1)
        if mouse_state["last_click"] is not None:
            click_xy = mouse_state["last_click"]
            mouse_state["last_click"] = None

            # Determine which detection was clicked
            idx = select_detection_from_click(detections, click_xy)
            if idx is not None:
                det = detections[idx]
                x1, y1, x2, y2 = det["bbox"]
                center = ((x1 + x2)/2, (y1 + y2)/2)

                # Update state
                state.primary_idx = idx
                state.primary_center = center
                state.primary_bbox = det["bbox"]
                state.selecting = False

                print(f"Primary fighter reassigned to index {idx}, center {center}")

def track_primary_bbox(
    prev_frame,
    curr_frame,
    prev_bbox,
    search_scale=2.0,
    min_confidence=0.4
):
    """
    Track the primary fighter using template matching between consecutive frames.
    Prints confidence score for debugging.
    """
    if prev_frame is None or prev_bbox is None:
        print("[TEMPLATE] Skipped: missing prev_frame or prev_bbox")
        return None

    x1, y1, x2, y2 = prev_bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    if bbox_w <= 0 or bbox_h <= 0:
        print("[TEMPLATE] Skipped: invalid bbox dimensions")
        return None

    # --- Extract template from previous frame ---
    template = prev_frame[y1:y2, x1:x2]
    if template.size == 0:
        print("[TEMPLATE] Skipped: empty template")
        return None

    h, w = curr_frame.shape[:2]

    # --- Define search window around previous center ---
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    sw = int(bbox_w * search_scale / 2)
    sh = int(bbox_h * search_scale / 2)

    sx1 = max(cx - sw, 0)
    sy1 = max(cy - sh, 0)
    sx2 = min(cx + sw, w)
    sy2 = min(cy + sh, h)

    search_region = curr_frame[sy1:sy2, sx1:sx2]

    if (
        search_region.shape[0] < bbox_h or
        search_region.shape[1] < bbox_w
    ):
        print("[TEMPLATE] Skipped: search region smaller than template")
        return None

    # --- Template matching ---
    res = cv2.matchTemplate(
        search_region,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print(f"[TEMPLATE] match confidence: {max_val:.3f}")

    if max_val < min_confidence:
        print(
            f"[TEMPLATE] Rejected (confidence {max_val:.3f} < {min_confidence})"
        )
        return None

    # --- Compute new bbox in full-frame coordinates ---
    new_x1 = sx1 + max_loc[0]
    new_y1 = sy1 + max_loc[1]
    new_x2 = new_x1 + bbox_w
    new_y2 = new_y1 + bbox_h

    return (new_x1, new_y1, new_x2, new_y2)
