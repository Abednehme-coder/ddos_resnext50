"""Unit tests for scripts/eval_primary_ckpt.py (no MindSpore)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_primary_ckpt import (  # noqa: E402
    is_primary_0p3_ckpt,
    primary_ckpt_path,
    require_primary_ckpt_for_eval,
)


def test_primary_ckpt_path_points_under_model_dir() -> None:
    p = primary_ckpt_path()
    assert p.name == "resnext50_32x4d_best.ckpt"
    assert p.parent.name == "per_second_0p3_focal_v2"


def test_is_primary_matches_canonical_layout() -> None:
    p = _REPO / "model" / "per_second_0p3_focal_v2" / "resnext50_32x4d_best.ckpt"
    assert is_primary_0p3_ckpt(p)


def test_is_primary_rejects_repo_root_alias() -> None:
    p = _REPO / "model" / "resnext50_32x4d_best.ckpt"
    assert not is_primary_0p3_ckpt(p)


def test_require_primary_accepts_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT", raising=False)
    require_primary_ckpt_for_eval(primary_ckpt_path())


def test_require_primary_rejects_non_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT", raising=False)
    bad = _REPO / "model" / "resnext50_32x4d_best.ckpt"
    with pytest.raises(SystemExit):
        require_primary_ckpt_for_eval(bad)


def test_require_primary_allowed_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT", "1")
    bad = _REPO / "model" / "resnext50_32x4d_best.ckpt"
    require_primary_ckpt_for_eval(bad)
