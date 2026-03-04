"""Tests for data router endpoints."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestYearValidation:
    """Tests for year validation."""

    def test_valid_year_format(self):
        """Test that valid years pass validation."""
        from app.routers.data import validate_year

        # Should not raise for valid years
        validate_year("2023")
        validate_year("2024")
        validate_year("2025")

    def test_invalid_year_format(self):
        """Test that invalid years raise HTTPException."""
        from app.routers.data import validate_year
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_year("23")
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info:
            validate_year("abcd")
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info:
            validate_year("1999")
        assert exc_info.value.status_code == 400


class TestFreeResponsesWithTags:
    """Tests for free responses with tags endpoint."""

    @pytest.mark.asyncio
    async def test_modified_tags_fallback(self):
        """Test that modified_tags falls back to original_tags when None."""
        from app.routers.data import get_free_responses_with_tags

        # Mock data where override exists but modified_tags is None
        mock_free_response = {
            "response_id": "resp_001",
            "survey_response_id": "survey_001",
            "question": "Q8",
            "question_type": "praise",
            "level": "Grammar",
            "response_text": "Great school!",
        }
        mock_tagging = {
            "id": "tag-1",
            "llm_tags": ["Academics"],
            "stability_score": 0.9,
            "tag_votes": {"Academics": 4},
            "keyword_mismatches": [],
            "dismissed": False,
            "dismissed_at": None,
        }
        mock_override = {
            "modified_tags": None,  # This is the bug case
        }

        with patch("app.routers.data.pb_client") as mock_pb:
            mock_pb.get_list = AsyncMock(return_value={
                "items": [mock_free_response],
                "page": 1,
                "perPage": 50,
                "totalItems": 1,
                "totalPages": 1,
            })
            mock_pb.get_first_list_item = AsyncMock(side_effect=[mock_tagging, mock_override])

            result = await get_free_responses_with_tags(
                year="2025",
                page=1,
                per_page=50,
            )

            # Should fall back to original_tags when modified_tags is None
            assert len(result.items) == 1
            assert result.items[0].tags == ["Academics"]


class TestSegmentStatistics:
    """Tests for segment statistics endpoint."""

    @pytest.mark.asyncio
    async def test_respondent_id_column_check(self):
        """Test that missing Respondent ID column raises proper error."""
        from app.routers.data import get_segment_statistics
        from fastapi import HTTPException

        # Create mock pipeline with data missing Respondent ID column
        mock_pipeline = MagicMock()
        mock_pipeline.loaded = True

        # Mock DataFrame without Respondent ID column
        mock_df = MagicMock()
        mock_df.columns = ["Other Column", "Level"]
        mock_df.height = 10
        mock_pipeline.data = mock_df
        mock_pipeline.config = {}

        # The filter function returns a filtered df
        mock_filtered = MagicMock()
        mock_filtered.columns = ["Other Column", "Level"]

        with patch("app.routers.data.pipeline_manager") as mock_pm:
            mock_pm.get_pipeline.return_value = mock_pipeline

            with patch("app.routers.data.DEMOGRAPHIC_SEGMENTS") as mock_segments:
                mock_segments.__contains__ = lambda self, key: key == "year_1"
                mock_segments.__getitem__ = lambda self, key: {
                    "name": "Year 1",
                    "inverse_name": "Not Year 1",
                    "filter": lambda df: mock_filtered,
                }

                # Should raise when inverse=True and Respondent ID missing
                with pytest.raises(HTTPException) as exc_info:
                    await get_segment_statistics("2025", "year_1", inverse=True)

                assert exc_info.value.status_code == 400
                assert "Respondent ID" in str(exc_info.value.detail)
