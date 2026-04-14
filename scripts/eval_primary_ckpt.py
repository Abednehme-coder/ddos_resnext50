"""
Primary checkpoint policy for evaluation and testing.

All standard eval scripts load only::

    model/per_second_0p3_focal_v2/resnext50_32x4d_best.ckpt

(trained on 0.3s time-window images). To load a different checkpoint (e.g. a
legacy experiment), set environment variable::

    HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT=1
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_ALLOW = "HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def primary_ckpt_path() -> Path:
    return repo_root() / "model" / "per_second_0p3_focal_v2" / "resnext50_32x4d_best.ckpt"


def primary_data_root() -> Path:
    return repo_root() / "dataset" / "images_per_second_window_0p3"


def is_primary_0p3_ckpt(ckpt: Path) -> bool:
    try:
        r = ckpt.resolve()
    except OSError:
        return False
    return r.name == "resnext50_32x4d_best.ckpt" and r.parent.name == "per_second_0p3_focal_v2"


def allow_non_primary_ckpt() -> bool:
    v = os.environ.get(_ENV_ALLOW, "").strip().lower()
    return v in ("1", "true", "yes")


def require_primary_ckpt_for_eval(ckpt: Path) -> None:
    if allow_non_primary_ckpt():
        return
    if is_primary_0p3_ckpt(ckpt):
        return
    primary = primary_ckpt_path()
    raise SystemExit(
        f"Evaluation and testing must use the 0.3s focal-v2 checkpoint:\n  {primary}\n"
        f"Got: {ckpt}\n"
        f"To override (not recommended), set {_ENV_ALLOW}=1"
    )
