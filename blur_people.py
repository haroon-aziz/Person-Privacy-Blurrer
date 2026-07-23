"""
Person Privacy Blurrer
Blur people in videos for privacy using Ultralytics ObjectBlurrer.

Usage:
    python blur_people.py --source input.mp4 --output output_blur.mp4
"""

import argparse

import cv2
from ultralytics import solutions


def main():
    parser = argparse.ArgumentParser(description="Blur people in a video for privacy.")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--output", default="output_blur.mp4", help="Path to output .mp4")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model weights")
    parser.add_argument("--blur-ratio", type=float, default=0.5, help="Blur intensity (0-1)")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold (lower catches more people)")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold (higher keeps overlapping people)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference size (higher detects smaller people)")
    parser.add_argument("--show", action="store_true", help="Show live preview window (not for Colab)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    assert cap.isOpened(), f"Error reading video: {args.source}"

    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    blurrer = solutions.ObjectBlurrer(
        model=args.model,
        blur_ratio=args.blur_ratio,
        classes=[0],  # class 0 = person
        show=args.show,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = blurrer.process(frame)
        video_writer.write(results.plot_im)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{n_frames} frames")

        if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
