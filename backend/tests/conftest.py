"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Sample test data
SAMPLE_TAXONOMY = {
    "tags": [
        {"name": "Academics", "keywords": ["curriculum", "learning", "education"]},
        {"name": "Teachers", "keywords": ["teacher", "staff", "faculty"]},
        {"name": "Communication", "keywords": ["communication", "updates", "inform"]},
    ]
}

SAMPLE_FREE_RESPONSES = [
    {
        "id": "test-1",
        "response_id": "resp_001",
        "year": "2025",
        "question": "Q8",
        "question_type": "praise",
        "level": "Grammar",
        "response_text": "The teachers are excellent and the curriculum is challenging.",
    },
    {
        "id": "test-2",
        "response_id": "resp_002",
        "year": "2025",
        "question": "Q9",
        "question_type": "improvement",
        "level": "Middle",
        "response_text": "Better communication about school events would be helpful.",
    },
]

SAMPLE_TAGGING_RESULTS = [
    {
        "id": "tag-1",
        "response_id": "resp_001",
        "year": "2025",
        "question": "Q8",
        "level": "Grammar",
        "response_text": "The teachers are excellent and the curriculum is challenging.",
        "llm_tags": ["Teachers", "Academics"],
        "tag_votes": {"Teachers": 4, "Academics": 3},
        "stability_score": 0.85,
        "keyword_mismatches": [],
        "review_status": None,
        "dismissed": False,
    },
    {
        "id": "tag-2",
        "response_id": "resp_002",
        "year": "2025",
        "question": "Q9",
        "level": "Middle",
        "response_text": "Better communication about school events would be helpful.",
        "llm_tags": ["Communication"],
        "tag_votes": {"Communication": 4},
        "stability_score": 0.95,
        "keyword_mismatches": [],
        "review_status": None,
        "dismissed": False,
    },
]


@pytest.fixture
def sample_taxonomy():
    """Provide sample taxonomy data."""
    return SAMPLE_TAXONOMY


@pytest.fixture
def sample_free_responses():
    """Provide sample free response data."""
    return SAMPLE_FREE_RESPONSES


@pytest.fixture
def sample_tagging_results():
    """Provide sample tagging result data."""
    return SAMPLE_TAGGING_RESULTS


@pytest.fixture
def mock_pb_client():
    """Create a mock PocketBase client."""
    mock = AsyncMock()

    # Setup common return values
    mock.get_list.return_value = {
        "items": [],
        "page": 1,
        "perPage": 50,
        "totalItems": 0,
        "totalPages": 0,
    }
    mock.get_full_list.return_value = []
    mock.create.return_value = {"id": "new-record-id"}
    mock.update.return_value = {"id": "updated-record-id"}
    mock.delete.return_value = True

    return mock


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock = MagicMock()

    # Mock chat completion response
    completion_mock = MagicMock()
    completion_mock.choices = [MagicMock()]
    completion_mock.choices[0].message.content = '["Teachers", "Academics"]'
    mock.chat.completions.create.return_value = completion_mock

    return mock


@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Create an async test client for async endpoint tests."""
    from app.main import app
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
