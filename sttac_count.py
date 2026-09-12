#!/usr/bin/env python3
"""Count physical apple defects by spatiotemporal trajectory association (STTAC)."""

import argparse
import csv
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sys
from typing import Iterable


DEFECT_NAMES = {1: "puncture", 2: "bruise", 3: "bitter pit"}
LOGGER = logging.getLogger("sttac.count")


@dataclass(frozen=True)
class STTACConfig:
    """Paper-reported physical and association parameters."""

    fps: float = 90.1
    roller_rpm: float = 120
    roller_radius_mm: float = 25
    apple_radius_mm: float = 40.25
    slip_ratio: float = 0.144
    delta_frames: int = 15
    min_frames: int = 5
    trim_ratio: float = 0.1
    y_threshold_px: float = 20

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.roller_rpm <= 0:
            raise ValueError("fps and roller_rpm must be positive")
        if self.roller_radius_mm <= 0 or self.apple_radius_mm <= 0:
            raise ValueError("roller and apple radii must be positive")
        if not 0 <= self.slip_ratio < 1:
            raise ValueError("slip_ratio must be in [0, 1)")
        if self.delta_frames < 0 or self.min_frames <= 0:
            raise ValueError("delta_frames must be non-negative and min_frames must be positive")
        if not 0 <= self.trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        if self.y_threshold_px < 0:
            raise ValueError("y_threshold_px must be non-negative")

    @property
    def half_rotation_seconds(self) -> float:
        return (
            30
            / (self.roller_rpm * (1 - self.slip_ratio))
            * (self.apple_radius_mm / self.roller_radius_mm)
        )

    @property
    def half_rotation_frames(self) -> int:
        return round(self.fps * self.half_rotation_seconds)

    @property
    def temporal_window(self) -> tuple[int, int]:
        center = self.half_rotation_frames
        return max(0, center - self.delta_frames), center + self.delta_frames


def load_tracking_results(path: Path) -> list[tuple[int, int, int, float, float]]:
    """Load frame, track, class, center-x, and center-y tuples from the tracking CSV."""
    rows = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            for line_number, fields in enumerate(reader, start=1):
                if not fields or all(not field.strip() for field in fields):
                    continue
                if line_number == 1 and fields[0].strip().lower() in {"frame_id", "帧号"}:
                    continue
                if len(fields) < 5:
                    raise ValueError(f"line {line_number} has {len(fields)} columns; expected at least 5")
                try:
                    rows.append(
                        (
                            int(fields[0]),
                            int(fields[1]),
                            int(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                        )
                    )
                except ValueError as exc:
                    message = f"line {line_number} contains a non-numeric trajectory value: {fields[:5]}"
                    raise ValueError(message) from exc
    except OSError as exc:
        raise OSError(f"Could not read tracking results: {path}") from exc
    return rows


def extract_segments(
    tracking_rows: Iterable[tuple[int, int, int, float, float]],
) -> list[dict]:
    """Group raw defect observations into ByteTrack trajectory segments by ID and class."""
    grouped: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for frame_id, track_id, class_id, center_x, center_y in tracking_rows:
        if class_id not in DEFECT_NAMES:
            continue
        grouped.setdefault((track_id, class_id), []).append((frame_id, center_x, center_y))

    segments = []
    for (track_id, class_id), points in grouped.items():
        ordered = sorted(points, key=lambda point: point[0])
        segments.append(
            {
                "original_track_id": track_id,
                "class_id": class_id,
                "start_frame": ordered[0][0],
                "end_frame": ordered[-1][0],
                "y_points": [point[2] for point in ordered],
            }
        )
    return segments


def trimmed_mean(values: Iterable[float], trim_ratio: float = 0.1) -> float:
    """Return the two-sided trimmed mean defined in the manuscript."""
    if not 0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("trimmed_mean requires at least one value")
    trim_count = math.floor(trim_ratio * len(ordered))
    trimmed = ordered[trim_count : len(ordered) - trim_count] if trim_count else ordered
    return sum(trimmed) / len(trimmed)


def compute_segment_features(segment: dict, config: STTACConfig) -> dict | None:
    """Compute F(S) after filtering segments shorter than L_min."""
    if len(segment["y_points"]) < config.min_frames:
        return None
    return {
        "class_id": segment["class_id"],
        "original_track_id": segment["original_track_id"],
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "y_ref": trimmed_mean(segment["y_points"], config.trim_ratio),
    }


def associate_segments(features: Iterable[dict], config: STTACConfig) -> dict[int, list[list[dict]]]:
    """Apply the manuscript's per-class greedy first-match association rule."""
    by_class = {class_id: [] for class_id in DEFECT_NAMES}
    for feature in features:
        if feature["class_id"] in by_class:
            by_class[feature["class_id"]].append(feature)

    window_start, window_end = config.temporal_window
    associated = {class_id: [] for class_id in DEFECT_NAMES}
    for class_id, class_features in by_class.items():
        for feature in sorted(class_features, key=lambda item: item["start_frame"]):
            matched_group = None
            for group in associated[class_id]:
                last_feature = group[-1]
                frame_gap = feature["start_frame"] - last_feature["end_frame"]
                y_gap = abs(feature["y_ref"] - last_feature["y_ref"])
                if window_start <= frame_gap <= window_end and y_gap <= config.y_threshold_px:
                    matched_group = group
                    break
            if matched_group is None:
                associated[class_id].append([feature])
            else:
                matched_group.append(feature)
    return associated


def count_defects(associated_groups: dict[int, list[list[dict]]]) -> tuple[dict[int, int], int]:
    """Count one physical defect for each final STTAC trajectory group."""
    counts = {class_id: len(associated_groups.get(class_id, [])) for class_id in DEFECT_NAMES}
    return counts, sum(counts.values())


def write_counts(path: Path, counts: dict[int, int], total: int) -> None:
    """Write category-specific and total defect counts."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["class_id", "defect_type", "count"])
            for class_id, name in DEFECT_NAMES.items():
                writer.writerow([class_id, name, counts[class_id]])
            writer.writerow(["total", "all defects", total])
    except OSError as exc:
        raise OSError(f"Could not write STTAC counts: {path}") from exc


def run_sttac(input_path: Path, output_path: Path, config: STTACConfig) -> tuple[dict[int, int], int]:
    tracking_rows = load_tracking_results(input_path)
    segments = extract_segments(tracking_rows)
    features = [compute_segment_features(segment, config) for segment in segments]
    valid_features = [feature for feature in features if feature is not None]
    groups = associate_segments(valid_features, config)
    counts, total = count_defects(groups)
    write_counts(output_path, counts, total)
    LOGGER.info(
        "STTAC processed %d rows, %d raw segments, and %d valid segments with temporal_window=%s",
        len(tracking_rows),
        len(segments),
        len(valid_features),
        config.temporal_window,
    )
    LOGGER.info("Counts=%s total=%d output=%s", counts, total, output_path)
    return counts, total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Trajectory CSV produced by postprocess.py")
    parser.add_argument("--output", type=Path, help="Output count CSV")
    parser.add_argument("--fps", type=float, default=90.1)
    parser.add_argument("--roller-rpm", type=float, default=120)
    parser.add_argument("--roller-radius-mm", type=float, default=25)
    parser.add_argument("--apple-radius-mm", type=float, default=40.25)
    parser.add_argument("--slip-ratio", type=float, default=0.144)
    parser.add_argument("--delta-frames", type=int, default=15)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--y-threshold-px", type=float, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DEBUG] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    args = build_parser().parse_args(argv)
    output_path = args.output or args.input.with_name(f"{args.input.stem}_sttac_counts.csv")
    try:
        config = STTACConfig(
            fps=args.fps,
            roller_rpm=args.roller_rpm,
            roller_radius_mm=args.roller_radius_mm,
            apple_radius_mm=args.apple_radius_mm,
            slip_ratio=args.slip_ratio,
            delta_frames=args.delta_frames,
            min_frames=args.min_frames,
            trim_ratio=args.trim_ratio,
            y_threshold_px=args.y_threshold_px,
        )
        run_sttac(args.input.expanduser().resolve(), output_path.expanduser().resolve(), config)
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.exception("STTAC counting failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
