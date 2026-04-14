"""Lightweight tests (no MindSpore)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_metrics():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "metrics_from_log.py"
    spec = importlib.util.spec_from_file_location("metrics_from_log", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_metrics_from_log_parses_line(tmp_path: Path) -> None:
    log = tmp_path / "d.log"
    log.write_text(
        json.dumps(
            {
                "pred": "ddos",
                "p_ddos": 0.7,
                "alert": True,
                "streak": 2,
            }
        )
        + "\n"
    )
    mm = _load_metrics()
    assert mm.summarize(log) == 0
