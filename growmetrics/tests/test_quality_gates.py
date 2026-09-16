"""
Unit tests for the 4-stage Quality Gate pipeline.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from app.generator.quality_gates import (
    GateError,
    gate1_json_schema,
    gate2_script_length,
    gate3_ast_safety,
    gate4_artifact_integrity,
)
from app.core.state_machine import FailureReason


# ── Gate 1 ────────────────────────────────────────────────────────────────

class TestGate1:
    def test_passes_valid_json(self):
        script = "word " * 80
        valid = f'{{"topic":"ph","title":"pH Scale","narration_script":"{script}","animation_code":"from manim import *\\nclass ChemistryScene(Scene): pass"}}'
        data = gate1_json_schema(valid)
        assert "topic" in data

    def test_passes_with_markdown_fences(self):
        raw = '```json\n{"topic":"a","title":"b","narration_script":"c","animation_code":"d"}\n```'
        data = gate1_json_schema(raw)
        assert data["topic"] == "a"

    def test_fails_invalid_json(self):
        with pytest.raises(GateError) as exc_info:
            gate1_json_schema("this is not json")
        assert exc_info.value.failure_reason == FailureReason.ERR_JSON_PARSE_FAILED

    def test_fails_missing_fields(self):
        raw = '{"topic": "ph", "title": "pH"}'  # missing narration_script and animation_code
        with pytest.raises(GateError) as exc_info:
            gate1_json_schema(raw)
        assert exc_info.value.failure_reason == FailureReason.ERR_JSON_SCHEMA_INVALID


# ── Gate 2 ────────────────────────────────────────────────────────────────

class TestGate2:
    LONG_SCRIPT = " ".join(["word"] * 80)  # 80 words

    def test_passes_valid_script_and_audio(self):
        gate2_script_length(self.LONG_SCRIPT, 35.0)  # should not raise

    def test_fails_short_word_count(self):
        short = " ".join(["word"] * 30)
        with pytest.raises(GateError) as exc_info:
            gate2_script_length(short, 35.0)
        assert exc_info.value.failure_reason == FailureReason.ERR_SCRIPT_TOO_SHORT

    def test_fails_audio_too_short(self):
        with pytest.raises(GateError) as exc_info:
            gate2_script_length(self.LONG_SCRIPT, 10.0)
        assert exc_info.value.failure_reason == FailureReason.ERR_AUDIO_TOO_SHORT

    def test_fails_audio_too_long(self):
        with pytest.raises(GateError) as exc_info:
            gate2_script_length(self.LONG_SCRIPT, 120.0)
        assert exc_info.value.failure_reason == FailureReason.ERR_AUDIO_TOO_LONG


# ── Gate 3: AST Safety ────────────────────────────────────────────────────

class TestGate3AST:
    VALID_CODE = "from manim import *\nclass ChemistryScene(Scene):\n    def construct(self): pass"

    def test_passes_valid_manim_code(self):
        gate3_ast_safety(self.VALID_CODE)

    def test_fails_syntax_error(self):
        with pytest.raises(GateError) as exc_info:
            gate3_ast_safety("def broken(: pass")
        assert exc_info.value.failure_reason == FailureReason.ERR_CODE_SYNTAX

    def test_fails_forbidden_exec(self):
        with pytest.raises(GateError) as exc_info:
            gate3_ast_safety("x = exec('import os')")
        assert exc_info.value.failure_reason == FailureReason.ERR_CODE_UNSAFE

    def test_fails_forbidden_os_import(self):
        with pytest.raises(GateError) as exc_info:
            gate3_ast_safety("import os\nos.system('rm -rf /')")
        assert exc_info.value.failure_reason == FailureReason.ERR_CODE_UNSAFE

    def test_fails_forbidden_subprocess(self):
        with pytest.raises(GateError) as exc_info:
            gate3_ast_safety("import subprocess")
        assert exc_info.value.failure_reason == FailureReason.ERR_CODE_UNSAFE


# ── Gate 4: Artifact Integrity ────────────────────────────────────────────

class TestGate4:
    def test_fails_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.mp4"
        with pytest.raises(GateError) as exc_info:
            gate4_artifact_integrity(missing, 30.0)
        assert exc_info.value.failure_reason == FailureReason.ERR_ARTIFACT_MISSING

    def test_fails_empty_file(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(GateError) as exc_info:
            gate4_artifact_integrity(empty, 30.0)
        assert exc_info.value.failure_reason == FailureReason.ERR_ARTIFACT_EMPTY

    def test_fails_no_ffprobe(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake data" * 100)
        with patch("subprocess.run", side_effect=FileNotFoundError("ffprobe not found")):
            with pytest.raises(GateError) as exc_info:
                gate4_artifact_integrity(video, 30.0)
            assert exc_info.value.failure_reason == FailureReason.ERR_RENDER_FAILED
