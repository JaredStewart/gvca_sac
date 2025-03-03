import logging

logger = logging.getLogger(__name__)

initial_headers = [
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

concluding_headers = [
    "Years at GVCA",
    "IEP, 504, ALP, or Read",
    "Minority",
]


levels = [
    "Grammar",
    "Middle",
    "High",
]

level_sequences = [
    ["Grammar"],
    ["Grammar", "Middle"],
    ["Grammar", "High"],
    ["Grammar", "Middle", "High"],
    ["Middle"],
    ["Middle", "High"],
    ["High"],
]

questions_for_each_school_level = [
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

has_free_response = [
    "What makes GVCA a good choice for you and your family?",
    "Please provide us with examples of how GVCA can better serve you and your family.",
]

has_generic_response = [
    "What makes GVCA a good choice for you and your family?",
    "Please provide us with examples of how GVCA can better serve you and your family.",
]

levels_of_satisfaction = [
    "Extremely Satisfied",
    "Satisfied",
    "Somewhat Satisfied",
    "Not Satisfied",
]
levels_of_reflection = [
    "Strongly Reflected",
    "Reflected",
    "Somewhat Reflected",
    "Not Reflected",
]
levels_of_effectiveness = [
    "Extremely Effective",
    "Effective",
    "Somewhat Effective",
    "Not Effective",
]
levels_of_welcoming = [
    "Extremely Welcoming",
    "Welcoming",
    "Somewhat Welcoming",
    "Not Welcoming",
]
question_responses = {
    questions_for_each_school_level[0]: levels_of_satisfaction,
    questions_for_each_school_level[1]: levels_of_satisfaction,
    questions_for_each_school_level[2]: levels_of_reflection,
    questions_for_each_school_level[3]: levels_of_satisfaction,
    questions_for_each_school_level[4]: levels_of_effectiveness,
    questions_for_each_school_level[5]: levels_of_effectiveness,
    questions_for_each_school_level[6]: levels_of_welcoming,
}

taxonomy_tags = [
    "Needs a Voice",
    "Concern",
    "All Around Support",
    "Curriculum",
    "Good Outcomes",
    "Policies/ Administration",
    "Teachers",
    "Culture/ Virtues",
    "Communication",
    "Community",
    "Extra-curriculars/ Sports",
    "Facilities",
    "No Improvement Listed",
]

taxonomy_string = ""
with open("taxonomy.md", "r") as f:
    taxonomy_string = f.read()


def make_input_headers():
    input_headers = initial_headers.copy()
    for level_sequence in level_sequences:
        for question in questions_for_each_school_level:
            for level in level_sequence:
                header = f"({level}) {question}"
                input_headers.append(header)
            if question in has_generic_response:
                header = f"(Generic) {question}"
                input_headers.append(header)
    input_headers.extend(concluding_headers)
    logger.debug(input_headers)
    return input_headers


input_headers = make_input_headers()


def to_parent_count(output_row, index_map):
    if (
        output_row[index_map["Submission Method"]]
        == "All parents and guardians will coordinate responses, and we will submit only one survey."
    ):
        return 2
    return 1


def to_response_empty(output_row, index_map):
    for question in questions_for_each_school_level:
        for level in levels:
            header = f"({level}) {question}"
            if output_row[index_map[header]] != "-":
                return False
        if question in has_generic_response:
            header = f"(Generic) {question}"
            if output_row[index_map[header]] != "-":
                return False
    return True


def to_combined_free_response(output_row, index_map):
    responses = []
    response_has_free_response = False
    for question in has_free_response:
        responses.append(f"{question}:")
        for level in levels:
            header = f"({level}) {question}"
            data = output_row[index_map[header]]
            if data != "-":
                response_has_free_response = True
                responses.append(f"- {data}")
        if question in has_generic_response:
            header = f"(Generic) {question}"
            data = output_row[index_map[header]]
            if data != "-":
                response_has_free_response = True
                responses.append(data)
    if not response_has_free_response:
        return "-"
    return "\n".join(responses)


additional_fields = {
    "N Parents Represented": to_parent_count,
    "Empty Response": to_response_empty,
}


def make_output_metadata():
    index_map = {}
    mapped_indexes = []
    output_headers = []
    for header in initial_headers:
        index_map[header] = len(output_headers)
        output_headers.append(header)

    for header in additional_fields:
        index_map[header] = len(output_headers)
        output_headers.append(header)

    for header in concluding_headers:
        index_map[header] = len(output_headers)
        output_headers.append(header)

    for question in questions_for_each_school_level:
        for level in levels:
            header = f"({level}) {question}"
            index_map[header] = len(output_headers)
            mapped_indexes.append(header)
            output_headers.append(header)
        if question in has_generic_response:
            header = f"(Generic) {question}"
            index_map[header] = len(output_headers)
            mapped_indexes.append(header)
            output_headers.append(header)

    logger.debug(output_headers)
    for key in mapped_indexes:
        logger.debug(key, index_map[key])

    return index_map, output_headers


index_map, output_headers = make_output_metadata()
