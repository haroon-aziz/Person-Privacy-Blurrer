"""
Person Privacy Blurrer — Webcam / Live Stream
Blur people in real time from a webcam (or any live stream URL).

Usage:
    python blur_webcam.py                    # default webcam (index 0)
    python blur_webcam.py --source 1         # second camera
    python blur_webcam.py --record out.mp4   # also save the blurred stream

Press "q" in the preview window to quit.
Note: requires a display — for headless/Colab use blur_people.py on a video file instead.
"""

import argparse

import cv2
from ultralytics import solutions


def main():
    parser = argparse.ArgumentParser(description="Blur people from a webcam in real time.")
    parser.add_argument("--source", default="0", help="Camera index (0, 1, ...) or stream URL")
    parser.add_argument("--record", default=None, help="Optional path to save the blurred stream as .mp4")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model weights")
    parser.add_argument("--blur-ratio", type=float, default=0.5, help="Blur intensity (0-1)")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (640 keeps webcam real-time)")
    args = parser.parse_args()

    # Camera index if numeric, otherwise treat as a stream URL
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f"Error opening camera/stream: {args.source}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    video_writer = None
    if args.record:
        video_writer = cv2.VideoWriter(
            args.record, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    blurrer = solutions.ObjectBlurrer(
        model=args.model,
        blur_ratio=args.blur_ratio,
        classes=[0],  # class 0 = person
        show=False,   # we handle display ourselves for the quit key
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    print("Running — press 'q' in the window to quit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = blurrer.process(frame)
        out = results.plot_im

        cv2.imshow("Person Privacy Blurrer (press q to quit)", out)
        if video_writer is not None:
            video_writer.write(out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Recording saved to {args.record}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
