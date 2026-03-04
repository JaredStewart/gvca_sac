"""Tests for tagging router endpoints."""

import pytest
from unittest.mock import patch, AsyncMock


class TestTagDistribution:
    """Tests for tag distribution endpoint."""

    @pytest.fixture
    def mock_tagging_results(self, sample_tagging_results):
        """Return sample tagging results for distribution tests."""
        return sample_tagging_results

    @pytest.mark.asyncio
    async def test_distribution_percentage_calculation(self, mock_tagging_results):
        """Test that percentages are calculated correctly using total tags, not responses."""
        from app.routers.tagging import get_tag_distribution

        # Mock pb_client.get_full_list
        with patch("app.routers.tagging.pb_client") as mock_pb:
            mock_pb.get_full_list = AsyncMock(return_value=mock_tagging_results)

            result = await get_tag_distribution("2025", None)

            # Total tags: Teachers(1) + Academics(1) + Communication(1) = 3
            # Teachers: 1/3 = 33.33%
            # Academics: 1/3 = 33.33%
            # Communication: 1/3 = 33.33%
            assert result["total_tags"] == 3
            assert result["unique_tags"] == 3

            # Check that percentages sum to ~100
            total_percentage = sum(item["percentage"] for item in result["distribution"])
            assert 99 <= total_percentage <= 101  # Allow for rounding

    @pytest.mark.asyncio
    async def test_distribution_empty_results(self):
        """Test distribution with no tagging results."""
        from app.routers.tagging import get_tag_distribution

        with patch("app.routers.tagging.pb_client") as mock_pb:
            mock_pb.get_full_list = AsyncMock(return_value=[])

            result = await get_tag_distribution("2025", None)

            assert result["total_tags"] == 0
            assert result["unique_tags"] == 0
            assert result["distribution"] == []


class TestTagToggle:
    """Tests for tag toggle endpoint."""

    @pytest.mark.asyncio
    async def test_toggle_adds_tag(self, mock_pb_client, sample_tagging_results):
        """Test that toggling with value=True adds a tag."""
        from app.routers.tagging import toggle_tag, TagToggleRequest

        with patch("app.routers.tagging.pb_client", mock_pb_client):
            # Setup: existing tagging result without "Safety" tag
            mock_pb_client.get_list.return_value = {
                "items": [sample_tagging_results[0]],
                "totalItems": 1,
            }
            mock_pb_client.update.return_value = {"id": "tag-1"}
            mock_pb_client.create.return_value = {"id": "override-1"}

            request = TagToggleRequest(tag="Safety", value=True)
            result = await toggle_tag("2025", "resp_001", request)

            assert "Safety" in result["tags"]
            assert result["value"] is True

    @pytest.mark.asyncio
    async def test_toggle_removes_tag(self, mock_pb_client, sample_tagging_results):
        """Test that toggling with value=False removes a tag."""
        from app.routers.tagging import toggle_tag, TagToggleRequest

        with patch("app.routers.tagging.pb_client", mock_pb_client):
            # Setup: existing tagging result with "Teachers" tag
            mock_pb_client.get_list.return_value = {
                "items": [sample_tagging_results[0]],
                "totalItems": 1,
            }
            mock_pb_client.update.return_value = {"id": "tag-1"}
            mock_pb_client.create.return_value = {"id": "override-1"}

            request = TagToggleRequest(tag="Teachers", value=False)
            result = await toggle_tag("2025", "resp_001", request)

            assert "Teachers" not in result["tags"]
            assert result["value"] is False
