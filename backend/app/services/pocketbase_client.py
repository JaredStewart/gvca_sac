"""PocketBase client for data persistence."""

import logging
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from app.config import get_settings

logger = logging.getLogger(__name__)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True for transient HTTP errors worth retrying."""
    if isinstance(exc, httpx.ConnectError | httpx.ReadTimeout | httpx.WriteTimeout):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (502, 503, 504)
    return False


# Retry decorator for transient PocketBase errors
_pb_retry = retry(
    retry=retry_if_exception(_is_transient_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    reraise=True,
)


class PocketBaseClient:
    """Client for interacting with PocketBase API.

    Uses a shared httpx.AsyncClient with connection pooling for efficiency
    and tenacity retries for transient network errors.
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.pocketbase_url
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client. Call on app shutdown."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _get_collection_url(self, collection: str) -> str:
        return f"/api/collections/{collection}/records"

    @_pb_retry
    async def create(self, collection: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new record in a collection."""
        client = self._get_client()
        response = await client.post(
            self._get_collection_url(collection),
            json=data,
        )
        if not response.is_success:
            logger.error(f"PocketBase create error: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json()

    @_pb_retry
    async def get_list(
        self,
        collection: str,
        page: int = 1,
        per_page: int = 50,
        filter_str: str | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Get paginated list of records."""
        params: dict[str, Any] = {"page": page, "perPage": per_page}
        if filter_str:
            params["filter"] = filter_str
        if sort:
            params["sort"] = sort

        client = self._get_client()
        response = await client.get(
            self._get_collection_url(collection),
            params=params,
        )
        response.raise_for_status()
        return response.json()

    @_pb_retry
    async def get_one(self, collection: str, record_id: str) -> dict[str, Any]:
        """Get a single record by ID."""
        client = self._get_client()
        response = await client.get(
            f"{self._get_collection_url(collection)}/{record_id}"
        )
        response.raise_for_status()
        return response.json()

    @_pb_retry
    async def update(
        self, collection: str, record_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Update an existing record."""
        client = self._get_client()
        response = await client.patch(
            f"{self._get_collection_url(collection)}/{record_id}",
            json=data,
        )
        if not response.is_success:
            logger.error(f"PocketBase update error: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json()

    @_pb_retry
    async def delete(self, collection: str, record_id: str) -> None:
        """Delete a record."""
        client = self._get_client()
        response = await client.delete(
            f"{self._get_collection_url(collection)}/{record_id}"
        )
        response.raise_for_status()

    async def get_first_list_item(
        self,
        collection: str,
        filter_str: str,
    ) -> dict[str, Any] | None:
        """Get the first record matching a filter, or None if not found."""
        result = await self.get_list(
            collection,
            page=1,
            per_page=1,
            filter_str=filter_str,
        )
        items = result.get("items", [])
        return items[0] if items else None

    async def get_full_list(
        self,
        collection: str,
        filter_str: str | None = None,
        sort: str | None = None,
        batch_size: int = 200,
    ) -> list[dict[str, Any]]:
        """Get all records from a collection (handles pagination)."""
        all_records = []
        page = 1

        while True:
            result = await self.get_list(
                collection,
                page=page,
                per_page=batch_size,
                filter_str=filter_str,
                sort=sort,
            )
            all_records.extend(result.get("items", []))

            if page >= result.get("totalPages", 1):
                break
            page += 1

        return all_records

    async def upsert(
        self,
        collection: str,
        data: dict[str, Any],
        filter_str: str,
    ) -> dict[str, Any]:
        """Insert or update a record based on filter."""
        # Try to find existing record
        existing = await self.get_list(collection, filter_str=filter_str, per_page=1)

        if existing.get("items"):
            # Update existing
            record_id = existing["items"][0]["id"]
            return await self.update(collection, record_id, data)
        else:
            # Create new
            return await self.create(collection, data)

    async def batch_create(
        self, collection: str, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create multiple records (not atomic)."""
        results = []
        client = self._get_client()
        for record in records:
            response = await client.post(
                self._get_collection_url(collection),
                json=record,
            )
            if response.status_code == 200:
                results.append(response.json())
        return results

    async def delete_by_filter(self, collection: str, filter_str: str) -> int:
        """Delete all records matching a filter."""
        records = await self.get_full_list(collection, filter_str=filter_str)
        deleted = 0
        client = self._get_client()
        for record in records:
            response = await client.delete(
                f"{self._get_collection_url(collection)}/{record['id']}"
            )
            if response.status_code == 204:
                deleted += 1
        return deleted

    # ========== Survey Responses Collection Methods ==========

    async def create_survey_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a survey response record."""
        if "imported_at" not in data:
            data["imported_at"] = datetime.utcnow().isoformat()
        return await self.create("survey_responses", data)

    async def create_survey_responses_batch(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create multiple survey response records."""
        now = datetime.utcnow().isoformat()
        for record in records:
            if "imported_at" not in record:
                record["imported_at"] = now
        return await self.batch_create("survey_responses", records)

    async def list_survey_responses(
        self,
        year: str,
        page: int = 1,
        per_page: int = 50,
        school_level: str | None = None,
        demographic: str | None = None,
        sort: str = "-imported_at",
    ) -> dict[str, Any]:
        """Get paginated survey responses with optional filters."""
        filters = [f'year = "{year}"']

        if school_level:
            filters.append(f'school_level ~ "{school_level}"')

        if demographic:
            # Handle demographic filter based on JSON field
            if demographic == "year1":
                filters.append('demographics.year1_family = true')
            elif demographic == "minority":
                filters.append('demographics.minority = true')
            elif demographic == "support":
                filters.append('demographics.support = true')
            elif demographic == "tenure_3plus":
                filters.append('demographics.tenure_years >= 3')

        filter_str = " && ".join(filters) if filters else None
        return await self.get_list(
            "survey_responses",
            page=page,
            per_page=per_page,
            filter_str=filter_str,
            sort=sort,
        )

    async def delete_survey_responses_by_year(self, year: str) -> int:
        """Delete all survey responses for a given year."""
        return await self.delete_by_filter("survey_responses", f'year = "{year}"')

    async def get_survey_response_count(self, year: str) -> int:
        """Get count of survey responses for a year."""
        result = await self.get_list(
            "survey_responses",
            filter_str=f'year = "{year}"',
            per_page=1,
        )
        return result.get("totalItems", 0)

    # ========== Free Responses Collection Methods ==========

    async def create_free_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a free response record."""
        return await self.create("free_responses", data)

    async def create_free_responses_batch(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create multiple free response records."""
        return await self.batch_create("free_responses", records)

    async def list_free_responses(
        self,
        year: str,
        page: int = 1,
        per_page: int = 50,
        question_type: str | None = None,
        level: str | None = None,
        question: str | None = None,
        sort: str = "response_id",
    ) -> dict[str, Any]:
        """Get paginated free responses with optional filters."""
        filters = [f'year = "{year}"']

        if question_type:
            filters.append(f'question_type = "{question_type}"')
        if level:
            filters.append(f'level = "{level}"')
        if question:
            filters.append(f'question ~ "{question}"')

        filter_str = " && ".join(filters)
        return await self.get_list(
            "free_responses",
            page=page,
            per_page=per_page,
            filter_str=filter_str,
            sort=sort,
        )

    async def get_free_response_by_id(self, response_id: str) -> dict[str, Any] | None:
        """Get a free response by its response_id."""
        result = await self.get_list(
            "free_responses",
            filter_str=f'response_id = "{response_id}"',
            per_page=1,
        )
        items = result.get("items", [])
        return items[0] if items else None

    async def delete_free_responses_by_year(self, year: str) -> int:
        """Delete all free responses for a given year."""
        return await self.delete_by_filter("free_responses", f'year = "{year}"')

    async def get_untagged_free_responses(
        self, year: str, batch_size: int = 200
    ) -> list[dict[str, Any]]:
        """Get free responses that don't have tagging results yet."""
        # Get all free responses for the year
        free_responses = await self.get_full_list(
            "free_responses",
            filter_str=f'year = "{year}"',
            batch_size=batch_size,
        )

        # Get all tagged response IDs
        tagging_results = await self.get_full_list(
            "tagging_results",
            filter_str=f'year = "{year}"',
            batch_size=batch_size,
        )
        tagged_ids = {r["response_id"] for r in tagging_results}

        # Return untagged responses
        return [r for r in free_responses if r["response_id"] not in tagged_ids]

    # ========== Tagging Results Collection Methods ==========

    async def get_tagging_results_for_review(
        self,
        year: str,
        stability_threshold: float = 0.75,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        """Get tagging results that need review (low stability or keyword mismatches)."""
        # Filter for responses needing review:
        # - stability_score < threshold OR keyword_mismatches is not empty
        # - NOT already approved or hidden
        filter_str = (
            f'year = "{year}" && '
            f'(stability_score < {stability_threshold} || keyword_mismatches != "[]") && '
            f'review_status != "approved" && review_status != "hidden"'
        )
        return await self.get_list(
            "tagging_results",
            page=page,
            per_page=per_page,
            filter_str=filter_str,
            sort="stability_score",  # Lowest stability first
        )

    async def update_review_status(
        self, record_id: str, status: str
    ) -> dict[str, Any]:
        """Update the review status of a tagging result."""
        return await self.update("tagging_results", record_id, {"review_status": status})

    # ========== Utility Methods ==========

    async def count(self, collection: str, filter_str: str | None = None) -> int:
        """Count records in a collection with optional filter."""
        result = await self.get_list(
            collection,
            page=1,
            per_page=1,
            filter_str=filter_str,
        )
        return result.get("totalItems", 0)

    async def get_distinct_years(self, collection: str) -> list[str]:
        """Get distinct years from a collection."""
        # PocketBase doesn't support DISTINCT, so we fetch all and dedupe in Python
        # This is efficient enough for small datasets like survey years
        records = await self.get_full_list(collection)
        years = sorted(set(r.get("year") for r in records if r.get("year")))
        return years


# Singleton instance
pb_client = PocketBaseClient()
