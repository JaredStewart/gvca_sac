"""OpenAI Batch API client for cost-effective bulk tagging."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.core.survey_config import get_taxonomy_tags
from app.core.tagging import build_system_prompt, build_user_prompt, compute_jaccard_stability
from app.core.transform import is_meaningful_response

logger = logging.getLogger(__name__)

# Token budget constants
TOKEN_BUDGET = 400_000     # Conservative: ~20% of OpenAI's 2M limit for reliable success
WORDS_TO_TOKENS = 2.5      # Conservative for structured prompts with taxonomy + JSON schema


class OpenAIBatchClient:
    """Client for interacting with OpenAI Batch API."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI()
        self.model = self.settings.default_llm_model
        self.n_samples = 4  # Number of samples per response for stability scoring

    def _build_request_body(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Build the request body for a batch API request.

        Includes structured output (json_schema) shared by all batch
        request builders.
        """
        from app.core.tagging import TagResponse

        return {
            "model": model,
            "n": self.n_samples,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tag_response",
                    "schema": TagResponse.model_json_schema(),
                },
            },
        }

    def _estimate_request_tokens(self, user_prompt: str, system_prompt: str) -> int:
        """Estimate tokens for a single request with n completions.

        Input tokens are counted once; output tokens (200 per completion)
        are counted n_samples times.
        """
        total_words = len(system_prompt.split()) + len(user_prompt.split())
        input_tokens = int(total_words * WORDS_TO_TOKENS)
        # Input once + max_tokens output per completion * n + JSON schema overhead
        return input_tokens + (200 * self.n_samples) + 100

    async def create_batch_input_file(
        self,
        responses: list[dict[str, Any]],
        model: str | None = None,
    ) -> str:
        """
        Create JSONL input file for batch processing.

        Each response becomes one request with n_samples completions
        (via the ``n`` parameter) for stability scoring.

        Returns: OpenAI File API ID
        """
        model_to_use = model or self.model
        system_prompt = build_system_prompt()
        lines = []

        for response in responses:
            response_id = response["response_id"]
            response_text = response["response_text"]
            question = response.get("question")

            # Defense-in-depth: skip non-meaningful responses (stale PB data)
            if not is_meaningful_response(response_text):
                continue

            user_prompt = build_user_prompt(response_text, question)

            request_body = {
                "custom_id": response_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self._build_request_body(
                    model_to_use, system_prompt, user_prompt,
                ),
            }
            lines.append(json.dumps(request_body))

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines))
            temp_path = f.name

        try:
            # Upload to OpenAI Files API
            with open(temp_path, "rb") as f:
                file_response = await self.client.files.create(
                    file=f,
                    purpose="batch",
                )

            logger.info(f"Created batch input file: {file_response.id} with {len(lines)} requests ({model_to_use})")
            return file_response.id
        finally:
            # Clean up temp file even if upload fails
            try:
                Path(temp_path).unlink()
            except OSError:
                pass

    async def create_batch_input_files(
        self,
        responses: list[dict[str, Any]],
        model: str | None = None,
    ) -> list[tuple[str, int, int]]:
        """
        Create multiple JSONL input files, splitting by estimated token budget.

        Splits responses into batches that fit within TOKEN_BUDGET to avoid
        exceeding OpenAI's enqueued token limit.

        Returns: list of (file_id, num_responses, estimated_tokens) per batch
        """
        model_to_use = model or self.model
        system_prompt = build_system_prompt()

        # Pre-compute prompts and estimate tokens per response
        response_data: list[tuple[dict[str, Any], str, int]] = []
        for response in responses:
            # Defense-in-depth: skip non-meaningful responses (stale PB data)
            if not is_meaningful_response(response["response_text"]):
                continue

            user_prompt = build_user_prompt(
                response["response_text"], response.get("question")
            )
            tokens_per_request = self._estimate_request_tokens(user_prompt, system_prompt)
            response_data.append((response, user_prompt, tokens_per_request))

        total_estimated = sum(t for _, _, t in response_data)
        logger.info(
            f"Estimated total tokens: {total_estimated:,} for {len(response_data)} responses "
            f"(n={self.n_samples} completions each, model={model_to_use})"
        )

        # Split into batches
        batches: list[list[tuple[dict[str, Any], str]]] = []
        batch_tokens: list[int] = []
        current_batch: list[tuple[dict[str, Any], str]] = []
        current_tokens = 0

        for response, prompt, tokens in response_data:
            if current_batch and current_tokens + tokens > TOKEN_BUDGET:
                batches.append(current_batch)
                batch_tokens.append(current_tokens)
                current_batch = []
                current_tokens = 0

            current_batch.append((response, prompt))
            current_tokens += tokens

        if current_batch:
            batches.append(current_batch)
            batch_tokens.append(current_tokens)

        logger.info(
            f"Split into {len(batches)} batch(es): "
            + ", ".join(f"{len(b)} responses (~{t:,} tokens)" for b, t in zip(batches, batch_tokens))
        )

        # Create and upload each batch file
        results: list[tuple[str, int, int]] = []
        for batch_idx, (batch, est_tokens) in enumerate(zip(batches, batch_tokens)):
            lines = []
            for response, user_prompt in batch:
                response_id = response["response_id"]
                request_body = {
                    "custom_id": response_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": self._build_request_body(
                        model_to_use, system_prompt, user_prompt,
                    ),
                }
                lines.append(json.dumps(request_body))

            # Write to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                f.write("\n".join(lines))
                temp_path = f.name

            try:
                with open(temp_path, "rb") as f:
                    file_response = await self.client.files.create(
                        file=f,
                        purpose="batch",
                    )

                logger.info(
                    f"Batch {batch_idx + 1}/{len(batches)}: "
                    f"file={file_response.id}, {len(batch)} responses, "
                    f"{len(lines)} requests, ~{est_tokens:,} tokens"
                )
                results.append((file_response.id, len(batch), est_tokens))
            finally:
                try:
                    Path(temp_path).unlink()
                except OSError:
                    pass

        return results

    async def submit_batch(self, input_file_id: str) -> dict[str, Any]:
        """
        Submit a batch job to OpenAI.

        Returns: dict with batch_id and status
        """
        batch = await self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "GVCA SAC Survey Tagging",
            },
        )

        logger.info(f"Submitted batch job: {batch.id} with status: {batch.status}")

        return {
            "batch_id": batch.id,
            "status": batch.status,
            "input_file_id": input_file_id,
            "created_at": batch.created_at,
        }

    async def check_batch_status(self, batch_id: str) -> dict[str, Any]:
        """
        Check status of a batch job.

        Returns: dict with status, progress, file IDs, and errors when available
        """
        batch = await self.client.batches.retrieve(batch_id)

        result = {
            "batch_id": batch.id,
            "status": batch.status,
            "request_counts": {
                "total": batch.request_counts.total if batch.request_counts else 0,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "failed": batch.request_counts.failed if batch.request_counts else 0,
            },
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
        }

        if batch.output_file_id:
            result["output_file_id"] = batch.output_file_id
        if batch.error_file_id:
            result["error_file_id"] = batch.error_file_id

        # Extract error details when available
        if batch.errors and batch.errors.data:
            result["errors"] = [
                {"code": e.code, "message": e.message}
                for e in batch.errors.data
            ]

        return result

    async def download_results(self, file_id: str) -> list[dict[str, Any]]:
        """
        Download and parse results from a batch output file.

        Returns: list of parsed results with custom_id and parsed tags
        """
        content = await self.client.files.content(file_id)
        lines = content.text.strip().split("\n")

        results = []
        for line in lines:
            if not line:
                continue

            try:
                data = json.loads(line)
                custom_id = data.get("custom_id", "")
                response = data.get("response", {})

                if response.get("status_code") == 200:
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        # Each choice is a separate completion (n parameter)
                        for choice in choices:
                            message = choice.get("message", {})
                            content = message.get("content", "[]")
                            # Parse structured output (TagResponse schema: {tags})
                            try:
                                parsed = json.loads(content)
                                if isinstance(parsed, dict) and "tags" in parsed:
                                    tags = parsed["tags"]
                                elif isinstance(parsed, list):
                                    tags = parsed
                                else:
                                    tags = []
                            except json.JSONDecodeError:
                                # Try to extract tags from non-JSON response
                                tags = self._extract_tags_from_text(content)

                            results.append({
                                "custom_id": custom_id,
                                "tags": tags,
                                "success": True,
                            })
                    else:
                        results.append({
                            "custom_id": custom_id,
                            "tags": [],
                            "success": False,
                            "error": "No choices in response",
                        })
                else:
                    results.append({
                        "custom_id": custom_id,
                        "tags": [],
                        "success": False,
                        "error": f"Status code: {response.get('status_code')}",
                    })

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing batch result line: {e}")
                continue

        return results

    def _extract_tags_from_text(self, text: str) -> list[str]:
        """Try to extract tag names from non-JSON formatted text."""
        import re
        from app.core.survey_config import get_taxonomy_tags

        # Look for quoted strings that might be tag names
        matches = re.findall(r'"([^"]+)"', text)

        if not matches:
            return []

        # Validate against taxonomy to filter out non-tag strings
        valid_tags = get_taxonomy_tags()
        return [m for m in matches if m in valid_tags]

    def aggregate_batch_results(
        self, results: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Aggregate results by response_id, computing stability scores.

        Multiple results per response_id come from the n completions
        returned per request. Groups by custom_id (= response_id) and
        counts tag votes across samples.

        Returns: dict mapping response_id to {llm_tags, tag_votes, stability_score}
        """
        # Group by response_id
        response_results: dict[str, list[list[str]]] = {}

        for result in results:
            if not result.get("success"):
                continue

            custom_id = result.get("custom_id", "")
            # custom_id is the response_id; legacy _s{n} suffix stripped for compat
            if "_s" in custom_id:
                response_id = custom_id.rsplit("_s", 1)[0]
            else:
                response_id = custom_id

            tags = result.get("tags", [])

            if response_id not in response_results:
                response_results[response_id] = []
            response_results[response_id].append(tags)

        # Compute stability scores
        aggregated = {}
        for response_id, tag_lists in response_results.items():
            # Count votes for each tag
            tag_votes: dict[str, int] = {}
            for tags in tag_lists:
                for tag in tags:
                    tag_votes[tag] = tag_votes.get(tag, 0) + 1

            n_samples = len(tag_lists)
            if n_samples == 0:
                continue

            # Filter to valid taxonomy tags only
            valid_tags = set(get_taxonomy_tags())
            tag_votes = {tag: votes for tag, votes in tag_votes.items() if tag in valid_tags}

            # Tags that appear in majority of samples
            threshold = n_samples / 2
            llm_tags = [tag for tag, votes in tag_votes.items() if votes > threshold]

            # Stability score: average pairwise Jaccard (IoU)
            stability_score = compute_jaccard_stability(tag_lists)

            aggregated[response_id] = {
                "llm_tags": llm_tags,
                "tag_votes": tag_votes,
                "stability_score": round(stability_score, 3),
                "n_samples": n_samples,
            }

        return aggregated

    async def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel a running batch job."""
        batch = await self.client.batches.cancel(batch_id)
        return {
            "batch_id": batch.id,
            "status": batch.status,
        }


# Singleton instance
openai_batch_client = OpenAIBatchClient()
