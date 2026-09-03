"""
tests/test_demo.py — Tests for demo & CLI entry points.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.demo import main as demo_main, run_demo
from app.main import main as app_main


class TestDemoMain:
    """Test CLI argument parsing and error exits."""

    def test_missing_image_fails(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["demo"])
        with pytest.raises(SystemExit) as exc:
            demo_main()
        assert exc.value.code != 0

    def test_invalid_image_path_returns_error(self, tmp_path):
        non_existent = str(tmp_path / "missing.jpg")
        ret = run_demo(image_path=non_existent, json_output=True)
        assert ret == 1

    def test_judge_mode_execution(self, tmp_path):
        test_file = tmp_path / "sample.jpg"
        test_file.write_bytes(b"dummy image data")

        with patch("app.demo.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {
                "image_path": str(test_file),
                "faces": [{"facial_area": {"x": 10, "y": 10, "w": 50, "h": 50}}],
                "face_result": None,
                "matches": [],
                "best_match": None,
                "fingerprint": None,
                "blockchain_record": None,
                "verification": None,
                "tamper_demo": None,
                "total_pipeline_duration_seconds": 0.42,
            }

            ret = run_demo(
                image_path=str(test_file),
                judge_mode=True,
                json_output=False,
            )
            assert ret == 0
            assert mock_pipeline.called
