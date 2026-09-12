"""Register paper-named GMED modules with the Ultralytics YAML model parser."""

from __future__ import annotations

import inspect
import textwrap

from modules.gmed import (
    CSPOmniKernel,
    ConvEdgeFusion,
    MultiScaleEdgeInfoGenerator,
    SPDConv,
    SelectScale,
    SkipConvEdgeFusion,
)


def _insert_before(source: str, marker: str, addition: str, description: str) -> str:
    if marker not in source:
        raise RuntimeError(f"The installed Ultralytics parser has no compatible {description} insertion point")
    return source.replace(marker, f"{addition}{marker}", 1)


def register_gmed_modules() -> None:
    """Add GMED classes and channel-inference rules to ``ultralytics.nn.tasks.parse_model``."""
    try:
        from ultralytics.nn import tasks
    except ImportError as exc:
        raise ImportError(
            "Ultralytics 8.4.41 is required. Install it before running GMED-YOLO."
        ) from exc

    if getattr(tasks, "_gmed_modules_registered", False):
        return

    tasks.SPDConv = SPDConv
    tasks.CSPOmniKernel = CSPOmniKernel
    tasks.ConvEdgeFusion = ConvEdgeFusion
    tasks.SkipConvEdgeFusion = SkipConvEdgeFusion
    tasks.MultiScaleEdgeInfoGenerator = MultiScaleEdgeInfoGenerator
    tasks.SelectScale = SelectScale

    source = textwrap.dedent(inspect.getsource(tasks.parse_model))
    spd_rule = """\
        elif m is SPDConv:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
"""
    source = _insert_before(source, "        elif m is AIFI:\n", spd_rule, "base-module")

    gmed_rules = """\
        elif m is CSPOmniKernel:
            c2 = ch[f]
            args = [c2]
        elif m in frozenset({ConvEdgeFusion, SkipConvEdgeFusion}):
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [[ch[x] for x in f], c2]
        elif m is MultiScaleEdgeInfoGenerator:
            c2 = [make_divisible(min(value, max_channels) * width, 8) for value in args[0]]
            args = [ch[f], c2]
        elif m is SelectScale:
            c2 = ch[f][args[0]]
"""
    tail_marker = "        else:\n            c2 = ch[f]\n"
    tail_index = source.rfind(tail_marker)
    if tail_index < 0:
        raise RuntimeError("The installed Ultralytics parser has no compatible custom-module insertion point")
    source = f"{source[:tail_index]}{gmed_rules}{source[tail_index:]}"

    try:
        exec(compile(source, "<gmed_parse_model>", "exec"), vars(tasks))
    except (SyntaxError, TypeError, ValueError) as exc:
        raise RuntimeError("Could not register GMED modules with the installed Ultralytics parser") from exc
    tasks._gmed_modules_registered = True
