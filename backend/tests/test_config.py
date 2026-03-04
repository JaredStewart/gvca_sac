"""Tests for configuration."""

import pytest


def test_survey_config_levels():
    """Test that LEVELS constant is correctly defined."""
    from app.core.survey_config import LEVELS

    assert "Grammar" in LEVELS
    assert "Middle" in LEVELS
    assert "High" in LEVELS
    assert len(LEVELS) >= 3


def test_survey_config_questions():
    """Test that QUESTIONS constant is defined."""
    from app.core.survey_config import QUESTIONS

    assert isinstance(QUESTIONS, list)
    assert len(QUESTIONS) > 0


def test_survey_config_free_response_questions():
    """Test that FREE_RESPONSE_QUESTIONS is a subset of QUESTIONS."""
    from app.core.survey_config import QUESTIONS, FREE_RESPONSE_QUESTIONS

    for q in FREE_RESPONSE_QUESTIONS:
        assert q in QUESTIONS, f"Free response question '{q}' not in QUESTIONS"
