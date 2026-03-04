"""Survey configuration and taxonomy definitions."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Headers and structure definitions
INITIAL_HEADERS = [
    "Respondent ID",
    "Collector ID",
    "Start",
    "End",
    "IP Address",
    "Email Address",
    "First Name",
    "Last Name",
    "Custom Data",
    "Submission Method",
    "Grade Selection",
]

CONCLUDING_HEADERS = [
    "Years at GVCA",
    "IEP, 504, ALP, or Read",
    "Minority",
]

LEVELS = ["Grammar", "Middle", "High"]

LEVEL_SEQUENCES = [
    ["Grammar"],
    ["Grammar", "Middle"],
    ["Grammar", "High"],
    ["Grammar", "Middle", "High"],
    ["Middle"],
    ["Middle", "High"],
    ["High"],
]

QUESTIONS = [
    "How satisfied are you with the education that Golden View Classical Academy provided this year?",
    "Given your children's education level at the beginning of of the year, how satisfied are you with their intellectual growth this year?",
    "GVCA emphasizes 7 core virtues: Courage, Moderation, Justice, Responsibility, Prudence, Friendship, and Wonder. How well is the school culture reflected by these virtues?",
    "How satisfied are you with your children's growth in moral character and civic virtue?",
    "How effective is the communication between your family and your children's teachers?",
    "How effective is the communication between your family and the school leadership?",
    "How welcoming is the school community?",
    "What makes GVCA a good choice for you and your family?",
    "Please provide us with examples of how GVCA can better serve you and your family.",
]

FREE_RESPONSE_QUESTIONS = [
    "What makes GVCA a good choice for you and your family?",
    "Please provide us with examples of how GVCA can better serve you and your family.",
]

# Response scales
SATISFACTION_SCALE = [
    "Extremely Satisfied",
    "Satisfied",
    "Somewhat Satisfied",
    "Not Satisfied",
]

REFLECTION_SCALE = [
    "Strongly Reflected",
    "Reflected",
    "Somewhat Reflected",
    "Not Reflected",
]

EFFECTIVENESS_SCALE = [
    "Extremely Effective",
    "Effective",
    "Somewhat Effective",
    "Not Effective",
]

WELCOMING_SCALE = [
    "Extremely Welcoming",
    "Welcoming",
    "Somewhat Welcoming",
    "Not Welcoming",
]

QUESTION_SCALES = {
    QUESTIONS[0]: SATISFACTION_SCALE,
    QUESTIONS[1]: SATISFACTION_SCALE,
    QUESTIONS[2]: REFLECTION_SCALE,
    QUESTIONS[3]: SATISFACTION_SCALE,
    QUESTIONS[4]: EFFECTIVENESS_SCALE,
    QUESTIONS[5]: EFFECTIVENESS_SCALE,
    QUESTIONS[6]: WELCOMING_SCALE,
}


def parse_taxonomy_from_file(filepath: str | Path) -> tuple[list[str], dict[str, list[str]], str]:
    """
    Parse taxonomy tags and keywords from markdown file.

    Args:
        filepath: Path to taxonomy.md file

    Returns:
        Tuple of (tag_names, tag_keywords, full_text)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"Taxonomy file not found: {filepath}")
        return [], {}, ""

    with open(filepath, "r") as f:
        taxonomy_text = f.read()

    # Extract tag names
    tags = re.findall(r"^###\s+(.+)$", taxonomy_text, flags=re.MULTILINE)
    tags = [tag.strip() for tag in tags]

    # Extract keywords for each tag
    tag_keywords: dict[str, list[str]] = {}
    current_tag = None
    in_keywords = False

    for line in taxonomy_text.split("\n"):
        line = line.strip()

        tag_match = re.match(r"^###\s+(.+)$", line)
        if tag_match:
            current_tag = tag_match.group(1).strip()
            tag_keywords[current_tag] = []
            in_keywords = False
            continue

        if "**Keywords:**" in line or "**keywords:**" in line:
            in_keywords = True
            continue

        if in_keywords and current_tag and line.startswith("- "):
            keyword = line[2:].strip()
            if keyword:
                tag_keywords[current_tag].append(keyword)
        elif in_keywords and line and not line.startswith("-") and not line.startswith("**"):
            in_keywords = False

    return tags, tag_keywords, taxonomy_text


def get_taxonomy() -> dict[str, list[str]]:
    """Get taxonomy keywords dictionary."""
    # Try multiple potential locations
    locations = [
        Path("taxonomy.md"),
        Path("../taxonomy.md"),
        Path("/app/taxonomy.md"),
        Path(__file__).parent.parent.parent.parent / "taxonomy.md",
    ]

    for loc in locations:
        if loc.exists():
            _, keywords, _ = parse_taxonomy_from_file(loc)
            return keywords

    logger.warning("Could not find taxonomy.md, using empty taxonomy")
    return {}


def get_taxonomy_tags() -> list[str]:
    """Get list of taxonomy tag names."""
    locations = [
        Path("taxonomy.md"),
        Path("../taxonomy.md"),
        Path("/app/taxonomy.md"),
        Path(__file__).parent.parent.parent.parent / "taxonomy.md",
    ]

    for loc in locations:
        if loc.exists():
            tags, _, _ = parse_taxonomy_from_file(loc)
            return tags

    return []


def get_taxonomy_string() -> str:
    """Get full taxonomy text."""
    locations = [
        Path("taxonomy.md"),
        Path("../taxonomy.md"),
        Path("/app/taxonomy.md"),
        Path(__file__).parent.parent.parent.parent / "taxonomy.md",
    ]

    for loc in locations:
        if loc.exists():
            _, _, text = parse_taxonomy_from_file(loc)
            return text

    return ""


@dataclass
class SurveyConfig:
    """Configuration for survey processing."""

    questions: list[str] = field(default_factory=lambda: QUESTIONS.copy())
    levels: list[str] = field(default_factory=lambda: LEVELS.copy())
    free_response_questions: list[str] = field(
        default_factory=lambda: FREE_RESPONSE_QUESTIONS.copy()
    )
    scales: dict[str, list[str]] = field(default_factory=lambda: QUESTION_SCALES.copy())

    @property
    def taxonomy_tags(self) -> list[str]:
        return get_taxonomy_tags()

    @property
    def taxonomy_keywords(self) -> dict[str, list[str]]:
        return get_taxonomy()

    @property
    def taxonomy_string(self) -> str:
        return get_taxonomy_string()

    def get_column_name(self, level: str, question: str) -> str:
        """Get DataFrame column name for a level/question combination."""
        return f"({level}) {question}"

    def get_scale_for_question(self, question: str) -> list[str]:
        """Get response scale for a question."""
        return self.scales.get(question, [])

    def is_free_response(self, question: str) -> bool:
        """Check if question is free response."""
        return question in self.free_response_questions
