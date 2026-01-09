from helpers import*
from state import*
def analyze_frame(
    frame,
    model,
    conf_threshold=0.3,
    draw_skeleton_flag=True,
    state: VideoAnalysisState = None
):
    """
    Main analyze-frame controller.
    """

    #use template tracking from previous frame
    tracked_bbox = None
    if state.prev_frame is not None:
        tracked_bbox = track_primary_bbox(prev_frame=state.prev_frame, curr_frame=frame,
                                       prev_bbox = state.primary_bbox)

    #detect fighters on frame
    detections = extract_detections(frame, model, conf_threshold)
    fighter_idx = None
    if state is not None:
        fighter_idx = get_primary_detection_idx(detections, state)
        # Update the state object
        if fighter_idx is not None:
            det = detections[fighter_idx]
            x1, y1, x2, y2 = det["bbox"]
            state.primary_idx = fighter_idx
            state.primary_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            state.primary_bbox = det["bbox"]

    #superimpose detections onto original frame
    output = render_detections(
        frame,
        detections,
        draw_skeleton_flag=draw_skeleton_flag,
        conf_thresh=0.3,
        fighter_idx=fighter_idx,
    )

    if tracked_bbox is not None:
        output = draw_tracked_bbox(
            output,
            tracked_bbox,
            color=(255, 0, 255),  # magenta
            thickness=2
        )

    return output, detections




