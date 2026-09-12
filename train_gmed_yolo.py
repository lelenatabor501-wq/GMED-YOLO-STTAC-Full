#!/usr/bin/env python3
"""Train GMED-YOLO with the detector settings reported in the manuscript."""

import argparse
import logging
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TRAIN_CONFIG = ROOT / "configs" / "train.yaml"

LOGGER = logging.getLogger("gmed_yolo.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="GMED-YOLO training configuration YAML",
    )
    parser.add_argument("--seed", type=int, default=0, help="Training seed")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional CUDA device override accepted by Ultralytics",
    )
    return parser


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def require_config_file(
    config: dict[str, object],
    key: str,
    description: str,
) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Training configuration must define a non-empty '{key}' path")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path

    return require_file(path, description)


def train(args: argparse.Namespace) -> None:
    config_yaml = require_file(args.config, "Training configuration YAML")

    # Import after CLI validation so `--help` works even before the environment is installed.
    from modules.gmed_registry import register_gmed_modules

    register_gmed_modules()
    from ultralytics import YOLO
    from ultralytics.utils import YAML

    config = YAML.load(str(config_yaml))
    if not isinstance(config, dict):
        raise ValueError(f"Training configuration must contain a YAML mapping: {config_yaml}")

    model_yaml = require_config_file(config, "model", "Model YAML")
    data_yaml = require_config_file(config, "data", "Dataset YAML")

    task = config.get("task", "detect")
    if task != "detect":
        raise ValueError(f"GMED-YOLO requires task='detect', but received task={task!r}")

    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        LOGGER.info(
            "Starting GMED-YOLO training with config=%s and seed=%d",
            config_yaml,
            args.seed,
        )
        model = YOLO(str(model_yaml), task=task)

        train_overrides: dict[str, object] = {
            "cfg": str(config_yaml),
            "data": str(data_yaml),
            "seed": args.seed,
        }
        if args.device is not None:
            train_overrides["device"] = args.device

        model.train(**train_overrides)
    finally:
        os.chdir(previous_cwd)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DEBUG] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    try:
        train(build_parser().parse_args(argv))
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        LOGGER.exception("GMED-YOLO training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
