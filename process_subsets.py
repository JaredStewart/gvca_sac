import csv
import re
from functools import partial
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI()


new_family_ids = {
    "114787513354",
    "114787162414",
    "114787146976",
    "114786145878",
    "114784927124",
    "114782645982",
    "114782513954",
    "114782459156",
    "114770821753",
    "114782392718",
    "114782321848",
    "114781117899",
    "114770118406",
    "114780239743",
    "114777946125",
    "114777826969",
    "114775543327",
    "114774977348",
    "114774883221",
    "114774687549",
    "114773338759",
    "114773060093",
    "114772690739",
    "114771236217",
    "114771029185",
    "114770826635",
    "114770793881",
    "114770485307",
    "114770472593",
    "114770360417",
    "114770355855",
    "114770271137",
    "114770216934",
    "114770134932",
    "114770096206",
    "114770118559",
    "114770095951",
    "114770085447",
}


def load_csv(file_path: str) -> List[Dict[str, str]]:
    """Load a CSV file with ISO-8859-1 encoding."""
    with open(file_path, encoding="ISO-8859-1", newline="") as csvfile:
        return list(csv.DictReader(csvfile))


def format_response(row: Dict[str, str]) -> str:
    """Format a survey response into a structured string."""
    response_type = "pro" if int(row["question_id"]) == 10 else "con"
    return f"- ({row['grade_level']}) {response_type}: {row['response']}"


def filter_by_keyword(row: Dict[str, str], keywords: List[str]) -> bool:
    """Return True if any keyword appears in the response text (case-insensitive)."""
    return any(kw.lower() in row["response"].lower() for kw in keywords)


def filter_by_regex(row: Dict[str, str], pattern: str) -> bool:
    """Return True if the regex pattern matches the response text."""
    return re.search(pattern, row["response"], re.IGNORECASE) is not None


def filter_new_family(row: Dict[str, str], new_family_ids: set) -> bool:
    """Filter for new family respondents based on their ID."""
    return row["respondent_id"] in new_family_ids


def filter_policies(row: Dict[str, str]) -> bool:
    """Check if Policies/Administration column has 'Yes'."""
    return row.get("Policies/ Administration", "").strip().lower() == "yes"


filters = {
    "virtu_charac": lambda row: filter_by_keyword(row, ["virtu", "charac"]),
    "classic": lambda row: filter_by_keyword(row, ["classic"]),
    "discipl": lambda row: filter_by_keyword(row, ["discipl"]),
    "bull": lambda row: filter_by_keyword(row, ["bull"]),
    "joy_fun": lambda row: filter_by_keyword(row, ["joy", "fun"]),
    "homework_stress_anx": lambda row: filter_by_regex(row, r"\b(home\s?work|stress|anxiety)\b"),
    "gradi_assign": lambda row: filter_by_keyword(row, ["gradi", "grades", "assign"]),
    "award": lambda row: filter_by_keyword(row, ["award"]),
    "tutor": lambda row: filter_by_keyword(row, ["tutor"]),
    "new_family": lambda row: filter_new_family(row, new_family_ids=new_family_ids),
    "not_understood": lambda row: filter_by_regex(
        row,
        r"\b(did(n't| not) know|unclear|confused|not sure|unsure|uncertain|wasn('t| not) explained|lack of communication|never heard|wasn('t| not) aware|did(n't| not) realize|more information needed|need (more|better) communication|not informed)\b",
    ),
    "broken": lambda row: filter_by_regex(
        row,
        r"\b(not working|broken|ineffective|does(n't| not) work|needs improvement|poorly (handled|implemented|executed)|lack of support|difficult to use|frustrating|system failure|unresponsive|process issue|too slow|no follow-up|ignored|lack of resources|overwhelmed|mismanaged|policy issue|bureaucracy|delays)\b",
    ),
    "challenging": lambda row: filter_by_regex(
        row,
        r"\b(bad|poor|disappointed|frustrated|unhappy|ineffective|terrible|needs improvement|struggling|difficult|challenging|confusing|lack of support|not enough|unhelpful|slow response|miscommunication|not listening|ignored|overwhelming|too much workload|stressful|not working|waste of time|low quality|unclear|complicated)\b",
    ),
    "policies": filter_policies,
}


def process_filter(data: List[Dict[str, str]], filter_key: str) -> List[str]:
    """Apply a filter and return formatted matching responses."""
    filter_func = filters.get(filter_key)
    if not filter_func:
        raise ValueError("Unknown filter key")
    matching_rows = [row for row in data if filter_func(row)]
    formatted = [format_response(row) for row in matching_rows]
    return formatted


def generate_baseline_prompt(keyword: str, responses: List[str]) -> str:
    """Generate a prompt for ChatGPT summarization."""
    str_responses = "\n".join(responses)
    return f"""
I want to take the following {len(responses)} survey free responses from a school satisfaction survey and produce up to five bullet points that soberly evaluate the 
data and explain common patterns.

Please note:
- This is a comparatively small set of responses from the survey which has 1041 total free response answers.
- Each bullet must not exceed 40 words.
- Avoid catastrophic language, overly judgmental tones, or recommendations.
- Clearly note if an observation applies only to a specific school level.
- Make certain your response generalizes across most responses, do not over specify on a single response.
- These responses were filtered using the keyword "{keyword}".

The responses are:
{str_responses}
    """.strip()


def generate_classical_ed_understanding_prompt(_: str, responses: List[str]) -> str:
    str_responses = "\n".join(responses)
    return f"""
Using the following {len(responses)} survey free responses from a school satisfaction survey.
Determine if parents understand what is entailed in a classical education, produce a series of bullet points which describe concepts parents understand well, and concepts
parents fail to adequately understand.

Please note:
- A classical education for the sake of this analysis can be defined as Classical education cultivates virtue through reasoned dialogue, deep study of literature, 
history, and science, and a structured, teacher-led classroom.
- This is a comparatively small set of responses from the survey which has 1041 total free response answers.
- Each bullet must not exceed 40 words.
- Avoid catastrophic language, overly judgmental tones, or recommendations.
- Clearly note if an observation applies only to a specific school level.
- Make certain your response generalizes across most responses, do not over specify on a single response.

The responses are:
{str_responses}
    """.strip()


def generate_new_families_response(_: str, responses: List[str]) -> str:
    str_responses = "\n".join(responses)
    return f"""
Using the following {len(responses)} survey free responses from a school satisfaction survey.
Determine which messages are common from new families, is the new family experience generally positive or negative?

Please note:
- This is a comparatively small set of responses from the survey which has 1041 total free response answers.
- Each bullet must not exceed 40 words.
- Avoid catastrophic language, overly judgmental tones, or recommendations.
- Clearly note if an observation applies only to a specific school level.
- Make certain your response generalizes across most responses, do not over specify on a single response.

The responses are:
{str_responses}
    """.strip()


def generate_response_function(question: str, _: str, responses: List[str]) -> str:
    str_responses = "\n".join(responses)
    return f"""
Using the following {len(responses)} survey free responses from a school satisfaction survey.
{question}

Please note:
- This is a comparatively small set of responses from the survey which has 1041 total free response answers.
- Each bullet must not exceed 40 words.
- Avoid catastrophic language, overly judgmental tones, or recommendations.
- Clearly note if an observation applies only to a specific school level.
- Make certain your response generalizes across most responses, do not over specify on a single response.

The responses are:
{str_responses}
    """.strip()


class BulletPoints(BaseModel):
    bullet_points: list[str]


def get_ai_summaries(prompt: str) -> List[Tuple[str, BulletPoints]]:
    models = [
        "gpt-4o-mini",
        "gpt-4o",
        # "o3-mini",
    ]

    prices = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.5, 10.0), "o3-mini": (1.1, 4.4)}

    responses = []
    total_price = 0
    for model in models:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            n=2,
            response_format=BulletPoints,
        )

        price = (
            prices[model][0] * completion.usage.prompt_tokens + prices[model][1] * completion.usage.completion_tokens
        ) / 1e6
        print(model, price)
        total_price += price
        for i, choice in enumerate(completion.choices):
            responses.append((f"{model} run {i + 1}", choice.message.parsed))
    print(total_price)

    return responses


def generate_markdown(
    category: str, keywords: str, responses: List[str], ai_summaries: List[Tuple[str, BulletPoints]]
) -> str:
    """Generate a structured markdown report."""
    str_ai_summary = ""
    for summary in ai_summaries:
        str_ai_summary += f"### {summary[0]}"
        for bullet in summary[1]:
            str_ai_summary += f"- {bullet}\n"
        str_ai_summary += "\n"
    str_responses = "\n".join(responses)
    return f"""
## Filtering Strategy
- **Category:** {category}
- **Keywords:** {keywords}
- **Matching Responses:** {len(responses)}

## AI Generated Summaries
{str_ai_summary}

## Extracted Responses
{str_responses}
    """.strip()


analyses = [
    ("Virtues", "virtue, character", "virtu_charac", partial(generate_baseline_prompt)),
    ("Classical Education", "classic", "classic", generate_classical_ed_understanding_prompt),
    ("Discipline", "discipline", "discipl", generate_baseline_prompt),
    ("Bullying", "bully, bullies", "bull", generate_baseline_prompt),
    ("Joy and Fun", "joy, fun", "joy_fun", generate_baseline_prompt),
    ("Homework Load", "homework, stress, anxiety", "homework_stress_anx", generate_baseline_prompt),
    ("Grading Policies", "grading, grades, assignments", "gradi_assign", generate_baseline_prompt),
    ("Awards", "award", "award", generate_baseline_prompt),
    ("Tutors", "tutor", "tutor", generate_baseline_prompt),
    ("What New Families are Saying", "n/a", "new_family", generate_new_families_response),
    (
        "Things Parents Don't Understand",
        "various",
        "not_understood",
        partial(
            generate_response_function,
            """
Identify what things parents fail to understand about the school.
Produce a list of bullet points describing the gaps in parent understanding.
""".strip(),
        ),
    ),
    (
        "Things Parents Think Don't Work",
        "various",
        "broken",
        partial(
            generate_response_function,
            """
Identify what things parents think are broken about the school.
Produce a list of bullet points describing the things parents think are broken.
""".strip(),
        ),
    ),
    (
        "Things Parents Find Challenging",
        "various",
        "challenging",
        partial(
            generate_response_function,
            """
Identify what things parents find challenging about the school.
Produce a list of bullet points describing the challenges parents regularly encounter
""".strip(),
        ),
    ),
    (
        "Thoughts about Administrative Policies",
        "various",
        "policies",
        partial(
            generate_response_function,
            """
Evaluate the pros and cons of the school's adminstrative policies described in the responses. 
Produce a list of bullet points which evaluate which policies are working well and which could use some refinement.
""".strip(),
        ),
    ),
]


if __name__ == "__main__":
    data = load_csv("processed/sara.csv")
    # For example, filtering on "bull" (bullying-related responses)
    f = open("summaries.md", "w")
    for analysis in analyses:
        topic_keyword = analysis[0]
        responses = process_filter(data, analysis[2])
        prompt = analysis[3](topic_keyword, responses)
        ai_summaries = get_ai_summaries(prompt)
        mkdown = generate_markdown(topic_keyword, analysis[1], responses, ai_summaries)
        f.write(mkdown)
        f.write("\n\n")
    f.close()
