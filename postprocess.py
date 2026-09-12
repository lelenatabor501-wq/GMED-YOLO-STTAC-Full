#!/usr/bin/env python3
"""Generate ByteTrack trajectories from a rotating-apple video using GMED-YOLO."""

import argparse
import csv
import logging
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TRACKER = ROOT / "configs" / "bytetrack.yaml"
DEFECT_CLASSES = {1: "puncture", 2: "bruise", 3: "bitter pit"}
CLASS_NAMES = {0: "apple", **DEFECT_CLASSES}
LOGGER = logging.getLogger("gmed_yolo.track")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained GMED-YOLO checkpoint")
    parser.add_argument("--source", type=Path, required=True, help="Input rotating-apple video")
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER, help="ByteTrack YAML")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "track", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.6, help="Detector confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="Detector NMS IoU threshold")
    parser.add_argument("--skip-first", type=int, default=0, help="Optional leading frames to omit")
    parser.add_argument("--skip-last", type=int, default=0, help="Optional trailing frames to omit")
    return parser


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def track(args: argparse.Namespace) -> tuple[Path, Path]:
    weights = require_file(args.weights, "GMED-YOLO checkpoint")
    source = require_file(args.source, "Input video")
    tracker = require_file(args.tracker, "ByteTrack configuration")
    if args.skip_first < 0 or args.skip_last < 0:
        raise ValueError("--skip-first and --skip-last must be non-negative")

    import cv2
    from modules.gmed_registry import register_gmed_modules

    register_gmed_modules()
    from ultralytics import YOLO

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / f"{source.stem}_trajectories.csv"
    video_path = output_dir / f"{source.stem}_tracked.mp4"

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {source}")

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_rate <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            f"Invalid video metadata for {source}: fps={frame_rate}, width={width}, height={height}"
        )

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {video_path}")

    model = YOLO(str(weights))
    frame_id = 0
    written_rows = 0
    try:
        with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(["frame_id", "track_id", "class_id", "center_x", "center_y"])

            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break
                frame_id += 1
                if frame_id <= args.skip_first:
                    continue
                if total_frames and frame_id > total_frames - args.skip_last:
                    break

                results = model.track(
                    frame,
                    tracker=str(tracker),
                    persist=True,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                )
                result = results[0]
                result.names = CLASS_NAMES
                writer.write(result.plot(font_size=1.5))

                boxes = result.boxes
                if boxes is None or boxes.id is None:
                    continue
                for index in range(len(boxes)):
                    class_id = int(boxes.cls[index].item())
                    if class_id not in DEFECT_CLASSES:
                        continue
                    track_id = int(boxes.id[index].item())
                    center_x, center_y, _, _ = boxes.xywh[index].cpu().tolist()
                    csv_writer.writerow([frame_id, track_id, class_id, f"{center_x:.3f}", f"{center_y:.3f}"])
                    written_rows += 1
    finally:
        capture.release()
        writer.release()

    LOGGER.info("Wrote %d defect trajectory rows to %s", written_rows, trajectory_path)
    LOGGER.info("Wrote annotated tracking video to %s", video_path)
    return trajectory_path, video_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DEBUG] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    try:
        track(build_parser().parse_args(argv))
    except (FileNotFoundError, ImportError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.exception("GMED-YOLO/ByteTrack trajectory generation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
