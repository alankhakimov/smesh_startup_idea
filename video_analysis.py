import cv2
import time
from analyze_frame import*

#main loop for running script
def main():
    video_path = "C:\Documents\Startup Idea\IMG_9264.mov"
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")
    conf_threshold = 0.2
    dim_background = True

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Retrieve FPS to control playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_duration = 1.0 / fps
    paused = False
    frame_idx = 0

    while True:
        if not paused:
            # Continuous playback
            ret, frame = cap.read()
            if not ret:
                break

            # --- Analyze frame here ---
            processed_frame = analyze_frame(frame, model,
                                            conf_threshold, dim_background)

            cv2.imshow("Video", processed_frame)
            frame_idx += 1

            # Check for key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                paused = True

            # Optional FPS enforcement
            time.sleep(frame_duration)

        else:
            # Step / slideshow mode
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("n"):
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = analyze_frame(frame, model,
                                                conf_threshold, dim_background)
                cv2.imshow("Video", processed_frame)
                frame_idx += 1
            elif key == ord("p"):
                # Previous frame (seek back)
                frame_idx = max(0, frame_idx - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    processed_frame = analyze_frame(frame, model,
                                                    conf_threshold, dim_background)
                    cv2.imshow("Video", processed_frame)
            elif key == ord(" "):
                paused = False
            elif key == ord("s"):
                paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()