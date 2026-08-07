import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app
import pytest


def test_app_builds_demo():
    assert app.demo is not None


def test_generation_requires_reference_audio():
    cloner = app.VoiceCloner.__new__(app.VoiceCloner)
    with pytest.raises(ValueError, match="reference audio"):
        cloner.generate(None, "Hello")


def test_generation_requires_text():
    cloner = app.VoiceCloner.__new__(app.VoiceCloner)
    with pytest.raises(ValueError, match="text"):
        cloner.generate("reference.wav", "  ")
