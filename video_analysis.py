import cv2
from ultralytics import YOLO
import time
from analyze_frame import*
from state import VideoAnalysisState

#main loop for running script
def main():
    video_path = "C:\Documents\Startup Idea\IMG_9264.mov"
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n-pose.pt")
    conf_threshold = 0.2

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Retrieve FPS to control playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_duration = 1.0 / fps

    state = VideoAnalysisState()

    # --- Create window BEFORE registering callback ---
    cv2.namedWindow("Video")

    # --- Mouse click storage ---
    mouse_state = {"last_click": None}

    cv2.setMouseCallback("Video", mouse_callback, mouse_state)

    # --- Initialize -> open first frame ---
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    state.paused = True
    state.selecting = True

    processed_frame, detections = analyze_frame(
        frame, model, conf_threshold, state=state
    )

    cv2.imshow("Video", processed_frame)
    state.prev_frame = frame.copy()
    print("Paused on first frame. Select a fighter to continue.")

    while state.selecting:
        cv2.imshow("Video", processed_frame)
        cv2.waitKey(1)

        if mouse_state["last_click"] is not None:
            click_xy = mouse_state["last_click"]

            idx = select_detection_from_click(detections, click_xy)

            mouse_state["last_click"] = None

            if idx is not None:
                det = detections[idx]
                x1, y1, x2, y2 = det["bbox"]
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # Update state
                state.primary_idx = idx
                state.primary_center = center
                state.primary_bbox = det["bbox"]
                state.selecting = False
                state.paused = False

                print(f"Clicked detection index: {idx}, center: {center}")

    # Main Playback -> display frame by frame
    while True:
        if not state.paused:
            # Continuous playback
            ret, frame = cap.read()
            if not ret:
                break
            #print(f"[DEBUG] primary_center before analyze_frame: {state.primary_center}")
            processed_frame, detections = analyze_frame(
                frame, model, conf_threshold, state = state
            )

            cv2.imshow("Video", processed_frame)
            state.frame_idx += 1
            state.prev_frame = frame.copy()

            # Key input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                state.paused = True

            # Optional FPS enforcement
            time.sleep(frame_duration)

        else:
            # Step / slideshow mode (blocking)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("n"):
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, detections = analyze_frame(
                    frame, model, conf_threshold, state = state
                )

                cv2.imshow("Video", processed_frame)
                state.frame_idx += 1
                state.prev_frame = frame.copy()

            elif key == ord("p"):
                # Previous frame (seek back)
                state.frame_idx = max(0, state.frame_idx - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, state.frame_idx)

                ret, frame = cap.read()
                if ret:
                    processed_frame, detections = analyze_frame(
                        frame, model, conf_threshold, state = state
                    )
                    cv2.imshow("Video", processed_frame)
                    state.prev_frame = frame.copy()

            elif key == ord(" "):
                state.paused = False

            elif key == ord("s"):
                state.paused = not state.paused
            elif key == ord("r"):
                # Manual reassignment triggered
                reassign_primary(detections, state, mouse_state)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()