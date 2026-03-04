"""AI-powered summarization for survey responses using OpenAI."""

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)


class SummaryResponse(BaseModel):
    """Schema for summarization response."""
    summary: str
    key_points: list[str]
    sentiment: str  # "positive", "negative", "mixed", "neutral"


async def summarize_responses(
    responses: list[dict[str, Any]],
    prompt_context: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Summarize a collection of survey responses using OpenAI.

    Args:
        responses: List of response objects with response_text field
        prompt_context: Optional context to focus the summary (e.g., "communication issues")
        model: OpenAI model to use

    Returns:
        Dictionary with summary, key_points, sentiment, and response_count
    """
    settings = get_settings()

    if not settings.openai_api_key:
        logger.warning("No OpenAI API key configured")
        return {
            "summary": "Unable to generate summary: No API key configured",
            "key_points": [],
            "sentiment": "neutral",
            "response_count": len(responses),
        }

    if not responses:
        return {
            "summary": "No responses to summarize",
            "key_points": [],
            "sentiment": "neutral",
            "response_count": 0,
        }

    # Build the prompt
    responses_text = []
    for i, r in enumerate(responses, 1):
        text = r.get("response_text", r.get("text", ""))
        level = r.get("level", "Unknown")
        question = r.get("question", "")
        if text:
            responses_text.append(f"{i}. [{level}] {text}")

    responses_block = "\n".join(responses_text)

    context_instruction = ""
    if prompt_context:
        context_instruction = f"\n\nFocus specifically on: {prompt_context}"

    system_prompt = """You are an expert at analyzing school survey feedback and summarizing parent concerns and praises.
You identify patterns, common themes, and the overall sentiment of the responses.
You are precise and provide actionable insights."""

    user_prompt = f"""Analyze and summarize these {len(responses)} survey responses from parents at a classical academy school.
{context_instruction}

RESPONSES:
{responses_block}

Provide a comprehensive summary that:
1. Identifies the main themes and concerns
2. Notes specific examples or patterns
3. Assesses the overall sentiment

Return your response as JSON with:
- "summary": A 2-4 sentence summary of the key themes and overall message
- "key_points": Array of 3-5 specific key points or themes (brief bullet points)
- "sentiment": One of "positive", "negative", "mixed", or "neutral"
"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            return {
                "summary": parsed.get("summary", "Unable to parse summary"),
                "key_points": parsed.get("key_points", []),
                "sentiment": parsed.get("sentiment", "neutral"),
                "response_count": len(responses),
            }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI response: {e}")
        return {
            "summary": "Error parsing AI response",
            "key_points": [],
            "sentiment": "neutral",
            "response_count": len(responses),
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI API error: {e}")
        return {
            "summary": f"API error: {e.response.status_code}",
            "key_points": [],
            "sentiment": "neutral",
            "response_count": len(responses),
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return {
            "summary": f"Error: {str(e)}",
            "key_points": [],
            "sentiment": "neutral",
            "response_count": len(responses),
        }
