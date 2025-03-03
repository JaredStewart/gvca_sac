import csv
import logging
import os
import textwrap
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import config

logger = logging.getLogger()

load_dotenv()
client = OpenAI()


# Existing tagging function that calls out to GPT and then filters tags
class SurveyFreeResponseEvaluation(BaseModel):
    tags: list[str]
    requests: list[str]


def process_free_response(text, n=4):
    label_response_prompt = f"""
    You are processing parent survey data from a school accountability committee. 
    The parents were asked to provide feedback about what is working well and what could be improved.

    You have a taxonomy of classifications to label free response data:
    {config.taxonomy_string}

    Evaluate the following input and extract:
    1. Relevant category tags.
    2. Concrete, specific requests.

    ```input
    {text}
    ```
    
    Return only valid tags from the taxonomy and concrete requests.
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": label_response_prompt},
        ],
        n=n,
        response_format=SurveyFreeResponseEvaluation,
    )

    # Post-process tags: Keep only those appearing in at least 3 completions
    tag_counts = Counter()
    requests = []

    for i in range(4):
        parsed_response = completion.choices[i].message.parsed
        requests.extend(parsed_response.requests)
        for tag in parsed_response.tags:
            tag_counts[tag] += 1

    kept_tags = [t for t, count in tag_counts.items() if count >= 3]

    return kept_tags, list(set(requests))  # Deduplicate requests


# New function that builds free response data with tagging.
def make_tagged_free_response_data(flattened_row):
    """
    Processes a flattened row of survey responses into multiple rows.
    For each free-response question (and level), we run the tagging model
    and add one binary column per taxonomy tag (yes if tagged, empty otherwise).
    """
    output_rows = []

    # Process each free response question by level
    for question in config.has_free_response:
        for level in config.levels:
            header = f"({level}) {question}"
            index = config.index_map[header]
            if flattened_row[index] != "-":
                # Base data: initial columns + metadata
                base_data = flattened_row[: len(config.initial_headers)]
                response_text = flattened_row[index]
                row_data = base_data + [question, level, response_text]

                # Process the free response text to get tags and requests.
                kept_tags, requests = process_free_response(response_text)

                # Create one column per taxonomy tag.
                # If the tag is present in kept_tags, put "yes", otherwise leave empty.
                tag_columns = []
                for tag in config.taxonomy:
                    tag_columns.append("yes" if tag in kept_tags else "")

                row_data.extend(tag_columns)
                output_rows.append(row_data)

        # Process generic responses if available
        if question in config.has_generic_response:
            header = f"(Generic) {question}"
            index = config.index_map[header]
            if flattened_row[index] != "-":
                base_data = flattened_row[: len(config.initial_headers)]
                response_text = flattened_row[index]
                row_data = base_data + [question, "Generic", response_text]

                kept_tags, requests = process_free_response(response_text)
                tag_columns = []
                for tag in config.taxonomy:
                    tag_columns.append("yes" if tag in kept_tags else "")

                row_data.extend(tag_columns)
                output_rows.append(row_data)

    return output_rows


def make_output_row(input_row):
    output_row = ["-"] * len(config.output_headers)
    for i, item in enumerate(input_row):
        if i < len(config.initial_headers):
            output_row[i] = input_row[i]
        elif item != "":
            input_header = config.input_headers[i]
            output_index = config.index_map[input_header]
            output_row[output_index] = item

    for field in config.additional_fields:
        output_row[config.index_map[field]] = config.additional_fields[field](output_row, config.index_map)

    return output_row


def transform_raw(filename):
    flattened_data = []
    free_response_data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) != len(config.input_headers):
                logger.error("input_length not matched")
            if i < 2:
                continue
            flattened_row = make_output_row(row)
            flattened_data.append(flattened_row)
            free_response_data.extend(make_tagged_free_response_data(flattened_row))

    return flattened_data, free_response_data


def compute_averages_and_weighted_totals(df, weight_by_parents=False):
    """
    Given a pandas dataframe with respondent data and survey responses,
    this function computes per-respondent averages for each school level
    (Grammar, Middle, High) as well as an overall average. It then computes
    the weighted averages for all respondents across these new columns.

    The function assumes:
      - config.questions_for_each_school_level is a list of survey questions.
      - config.levels is a list of school levels, e.g. ["Grammar", "Middle", "High"].
      - config.has_free_response is a list of questions to ignore for numeric analysis.
      - config.question_responses is a dict mapping each question to its list of allowed responses,
          where the rank is computed as: rank_value = 4 - (index in allowed responses).

    Parameters:
      df (pd.DataFrame): The dataframe with survey data.
      weight_by_parents (bool): If True, the weighted average is computed using the
                                'N Parents Represented' column as weights.

    Returns:
      tuple: (decorated_df, weighted_totals) where decorated_df is the input dataframe
             with new columns for each average, and weighted_totals is a dictionary
             containing the weighted averages for "Overall Average", "Grammar Average",
             "Middle Average", and "High Average".
    """

    def compute_row_averages(row):
        overall_sum, overall_count = 0, 0
        grammar_sum, grammar_count = 0, 0
        middle_sum, middle_count = 0, 0
        high_sum, high_count = 0, 0

        # Loop over each question in the configuration.
        for question in config.questions_for_each_school_level:
            # Skip questions that are free-response.
            if question in config.has_free_response:
                continue

            # Get allowed responses for this question.
            allowed_responses = config.question_responses.get(question, [])
            # Loop over each school level.
            for level in config.levels:
                # Construct the column name as in the dataframe.
                col_name = f"({level}) {question}"
                # If the column is missing, skip.
                if col_name not in row:
                    continue
                cell = row[col_name]
                # Skip if the cell is empty or marked as a non-response.
                if pd.isnull(cell) or cell == "-":
                    continue
                try:
                    # Find the index of the response in the allowed list.
                    response_index = allowed_responses.index(cell)
                except ValueError:
                    # If the response isn’t recognized, skip.
                    print("response not tracked")
                    continue

                # Compute rank value based on the allowed ordering.
                rank_value = 4 - response_index

                # Update overall accumulators.
                overall_sum += rank_value
                overall_count += 1

                # Update level-specific accumulators.
                if level == "Grammar":
                    grammar_sum += rank_value
                    grammar_count += 1
                elif level == "Middle":
                    middle_sum += rank_value
                    middle_count += 1
                elif level == "High":
                    high_sum += rank_value
                    high_count += 1

        # Compute averages for the row (if at least one valid response was found).
        row["Overall Average"] = overall_sum / overall_count if overall_count > 0 else None
        row["Grammar Average"] = grammar_sum / grammar_count if grammar_count > 0 else None
        row["Middle Average"] = middle_sum / middle_count if middle_count > 0 else None
        row["High Average"] = high_sum / high_count if high_count > 0 else None

        return row

    # Apply the row-wise function to decorate the dataframe with new average columns.
    df = df.apply(compute_row_averages, axis=1)

    # Determine weights: either use the "N Parents Represented" column (converted to numeric)
    # or use a default weight of 1 for each respondent.
    if weight_by_parents:
        weights = pd.to_numeric(df["N Parents Represented"], errors="coerce").fillna(0)
    else:
        weights = pd.Series(1, index=df.index)

    def weighted_avg(column_name):
        valid = df[column_name].notna()
        if valid.sum() == 0:
            return np.nan
        return (df.loc[valid, column_name] * weights[valid]).sum() / weights[valid].sum()

    # Compute the weighted average for each of the four new columns.
    weighted_overall = weighted_avg("Overall Average")
    weighted_grammar = weighted_avg("Grammar Average")
    weighted_middle = weighted_avg("Middle Average")
    weighted_high = weighted_avg("High Average")

    weighted_totals = {
        "Overall Average": weighted_overall,
        "Grammar Average": weighted_grammar,
        "Middle Average": weighted_middle,
        "High Average": weighted_high,
    }

    return df, weighted_totals


def write_flattened_file(filename, flattened_data):
    output_rows = []
    output_rows.append(config.output_headers)
    output_rows.extend(flattened_data)
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)


# Updated file writing function that now includes the taxonomy tag columns in the header.
def write_free_response_file(filename, free_response_data):
    output_rows = []
    headers = config.initial_headers.copy()
    headers.extend(["Question", "Level", "Response"])
    # Add one header per taxonomy tag.
    headers.extend(config.taxonomy_tags)
    output_rows.append(headers)

    output_rows.extend(free_response_data)
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)


def raw_to_processed(year: str):
    base_file = f"{year}.csv"
    flattened_data, free_response_data = transform_raw(os.path.join("data", base_file))
    write_flattened_file(os.path.join("processed", base_file), flattened_data)
    write_free_response_file(os.path.join("processed", f"free_response_{base_file}"), free_response_data)


def load_flattened(year: str):
    df = pd.read_csv(f"processed/{year}.csv")
    df = df[~df["Empty Response"]].replace("-", pd.NA)
    df["Start"] = pd.to_datetime(df["Start"], format="%m/%d/%Y %I:%M:%S %p")
    df["End"] = pd.to_datetime(df["End"], format="%m/%d/%Y %I:%M:%S %p")
    return df


def plot_start_end_times(df):
    plt.figure(figsize=(10, 5))
    plt.hist(df["Start"], bins=20, alpha=0.5, label="Start", color="blue")
    plt.hist(df["End"], bins=20, alpha=0.5, label="End", color="red")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Distribution of Start and End Times")
    plt.xticks(rotation=45)
    plt.savefig(
        "artifacts/2025 Start and End Time Distribution",
        transparent=True,
    )
    return plt


def plot_distribution_of_durations(df):
    # Compute duration in hours
    df["Duration"] = (df["End"] - df["Start"]).dt.total_seconds() / 3600  # Convert to hours

    # Define duration buckets
    bins = [
        0,
        5 / 60,
        10 / 60,
        0.25,
        0.5,
        1,
        6,
        24,
        168,
        float("inf"),
    ]  # Hours: 30min, 1hr, 6hrs, 1 day, 1 week, 7+ days
    labels = ["<5 min", "5-10 min", "10-15 min", "15-30 min", "30-60 min", "1-6 hrs", "6-24 hrs", "1-7 days", "7+ days"]
    df["Duration Category"] = pd.cut(df["Duration"], bins=bins, labels=labels, right=False)

    # Plot Histogram for Durations
    plt.figure(figsize=(8, 5))
    df["Duration Category"].value_counts().sort_index().plot(kind="bar", color="orange")
    plt.xlabel("Duration Category")
    plt.ylabel("Count")
    plt.title("Distribution of Time Taken (Start to End)")
    plt.xticks(rotation=45)
    plt.savefig(
        "artifacts/2025 Duration Distribution",
        transparent=True,
    )
    return plt


def calculate_question_totals(df_, weight_by_parents: bool):
    results = []
    filters = {
        "Year 1 Families": pd.to_numeric(df_["Years at GVCA"]) == 1,
        "Not Year 1 Families": pd.to_numeric(df_["Years at GVCA"]) > 1,
        "Year 3 or Less Families": pd.to_numeric(df_["Years at GVCA"]) <= 3,
        "Year 4 or More Families": pd.to_numeric(df_["Years at GVCA"]) > 3,
        "Minority": df_["Minority"] == "Yes",
        "Not Minority": df_["Minority"] != "Yes",
        "Support": df_["IEP, 504, ALP, or Read"] == "Yes",
        "Not Support": df_["IEP, 504, ALP, or Read"] != "Yes",
    }

    for question in config.questions_for_each_school_level:
        response_levels = config.question_responses.get(question, [])

        for response in response_levels:
            response_data = {"Question": question, "Response": response}

            schoolwide_counts, schoolwide_total = _calculate_totals(
                df_, question, response, config.levels, weight_by_parents
            )
            response_data.update(_format_counts_and_percentages("total", schoolwide_counts, schoolwide_total, response))

            for level in config.levels:
                level_counts, level_total = _calculate_totals(df_, question, response, [level], weight_by_parents)
                response_data.update(_format_counts_and_percentages(level, level_counts, level_total, response))

            for filter_name, filter_condition in filters.items():
                filtered_counts, filtered_total = _calculate_totals(
                    df_[filter_condition], question, response, config.levels, weight_by_parents
                )
                response_data.update(
                    _format_counts_and_percentages(filter_name, filtered_counts, filtered_total, response)
                )

            results.append(response_data)

    return pd.DataFrame(results)


def _calculate_totals(df_, question, response, levels, weight_by_parents):
    """Helper to calculate counts and totals for given levels."""
    totals = {}
    overall_total = 0

    for level in levels:
        column_name = f"({level}) {question}"
        if column_name in df_.columns:
            filtered_df = df_[df_[column_name] == response]

            if weight_by_parents:
                response_sum = filtered_df["N Parents Represented"].astype(float).sum()
                level_total = df_[~df_[column_name].isna()]["N Parents Represented"].astype(float).sum()
            else:
                response_sum = len(filtered_df)
                level_total = len(df_[column_name].dropna())

            totals[response] = totals.get(response, 0) + response_sum
            overall_total += level_total

    return totals, overall_total


def _format_counts_and_percentages(label, counts, total, response):
    """Helper to format counts and percentages for a given response."""
    count = counts.get(response, 0)
    percentage = (count / total) * 100 if total > 0 else 0
    return {f"N_{label}": count, f"%_{label}": percentage}


def calculate_top_two_from_rollup(rolled_up_data):
    results = []

    for question in config.questions_for_each_school_level:
        top_two_responses = config.question_responses.get(question, [])[:2]  # Get first two satisfaction levels

        # Filter the rolled-up data for relevant responses
        filtered_data = rolled_up_data[(rolled_up_data["Question"] == question)]
        # ()

        if filtered_data.empty:
            continue

        response_data = {"Question": question}

        # Aggregate across all relevant columns (e.g., total, school levels, and filters)
        for column in rolled_up_data.columns:
            if column.startswith("N_"):  # Sum counts for relevant responses
                total_count = filtered_data[column].sum()
                total_responses = filtered_data[filtered_data["Response"].isin(top_two_responses)][column].sum()

                response_data[column] = total_responses
                response_data[column.replace("N_", "%_")] = (
                    (total_responses / total_count) * 100 if total_responses > 0 else 0
                )

        results.append(response_data)

    return pd.DataFrame(results)


def create_stacked_bar_chart(
    filename: str,
    title: str,
    x_axis_label: str,
    x_data_labels: list,
    proportions: dict,
    data_keys: list = None,
    savefig: bool = False,
    subfolder="artifacts",
) -> None:
    """
    Create and save a stacked bar chart.

    Parameters:
      title: The title of the chart.
      x_axis_label: Label for the x-axis.
      x_data_labels: Labels for the x-axis ticks.
      proportions: A dict where each key maps to a list of proportions for each response option.
      data_keys: Optional list specifying the order in which to read from proportions.
                 If None, defaults to non–free-response questions in config.
      savefig: If True, saves the figure.
      subfolder: Folder to save the figure.
    """
    # If no custom keys are provided, default to all non–free-response questions.
    if data_keys is None:
        data_keys = [q for q in config.questions_for_each_school_level if q not in config.has_free_response]

    # Extract segments for the stacked bar.
    # We assume that each value in proportions is a list with 4 elements corresponding to:
    # index 0: "Very", index 1: "Satisfied", index 2: "Somewhat", index 3: "Not"
    r1 = [proportions[key][3] for key in data_keys]
    r2 = [proportions[key][2] for key in data_keys]
    r3 = [proportions[key][1] for key in data_keys]
    r4 = [proportions[key][0] for key in data_keys]

    fig, ax = plt.subplots(1, figsize=(20, 8))
    ax.bar(
        x_data_labels,
        r4,
        label="Very",
        color="#6caf40",
        bottom=[q1 + q2 + q3 for q1, q2, q3 in zip(r1, r2, r3)],
    )
    ax.bar(
        x_data_labels,
        r3,
        label="Satisfied",
        color="#4080af",
        bottom=[q1 + q2 for q1, q2 in zip(r1, r2)],
    )
    ax.bar(x_data_labels, r2, label="Somewhat", color="#f6c100", bottom=r1)
    ax.bar(x_data_labels, r1, label="Not", color="#ae3f3f")

    ax.set_title(title)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Proportion")

    # Adjust the position to make room for the legend.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if savefig:
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        print("saving")
        plt.savefig(
            f"{subfolder}/{filename}.png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.5,
        )
    return plt


def to_proportions_and_labels(df, col):
    print(col)
    response_proportions = (
        df.groupby(["Question", "Response"])[col].sum().unstack(fill_value=0)  # Pivot so that each response is a column
    )

    # Normalize by row sum to get proportions
    response_proportions = response_proportions.div(response_proportions.sum(axis=1), axis=0)

    proportions = {}
    labels = []
    for question in config.questions_for_each_school_level:
        score = 0
        if question in config.has_free_response:
            continue
        proportions[question] = []
        n_options = len(config.question_responses.get(question, []))
        for i, response in enumerate(config.question_responses.get(question, [])):
            proportion = response_proportions.loc[question, response]
            proportions[question].append(proportion)
            score += proportion * (n_options - i)
        labels.append(f"{textwrap.fill(question, 35)}\n({score:.2f})")

    return proportions, labels


def plot_sequence(year, grouping, df_, savefig=False):
    splits = [
        ("All Responses", "N_total"),
        ("Grammar Responses", "N_Grammar"),
        ("Middle Responses", "N_Middle"),
        ("High Responses", "N_High"),
        ("Minority Responses", "N_Minority"),
        ("Support Responses", "N_Support"),
    ]

    for split in splits:
        proportions, labels = to_proportions_and_labels(df_, split[1])
        create_stacked_bar_chart(
            filename=f"{year} {grouping} {split[0]}",
            title=f"{year} {grouping} {split[0]}",
            x_axis_label="Response Summary",
            x_data_labels=labels,
            proportions=proportions,
            savefig=savefig,
        )


def plot_individual_question_stacked_bars(year: str, rolled_up_data, savefig: bool = False):
    plts = []
    groups = ["Whole School", "Grammar", "Middle", "High"]

    for i, question in enumerate(config.questions_for_each_school_level):
        if question in config.has_free_response:
            continue

        responses = config.question_responses.get(question, [])
        # Build a proportions dictionary with keys corresponding to each group.
        # Each value is a list of percentages in the order of responses.
        proportions_for_question = {}
        for group in groups:
            percentages = []
            for response in responses:
                row = rolled_up_data[
                    (rolled_up_data["Question"] == question) & (rolled_up_data["Response"] == response)
                ]
                if not row.empty:
                    if group == "Whole School":
                        percentages.append(row.iloc[0].get("%_total", 0))
                    elif group == "Grammar":
                        percentages.append(row.iloc[0].get("%_Grammar", 0))
                    elif group == "Middle":
                        percentages.append(row.iloc[0].get("%_Middle", 0))
                    elif group == "High":
                        percentages.append(row.iloc[0].get("%_High", 0))
                else:
                    percentages.append(0)
            proportions_for_question[group] = percentages

        # Create x-axis labels that include each group along with its computed score.
        x_labels = []
        for group in groups:
            percentages = proportions_for_question[group]
            # Compute the score using the ranking rule: rank_value = 4 - response_index.
            # Divide by 100 to convert percentage to a proportion.
            score = sum((p / 100) * (4 - j) for j, p in enumerate(percentages))
            x_labels.append(f"{group}\n({score:.2f})")

        # Use create_stacked_bar_chart to plot this single question.
        plt = create_stacked_bar_chart(
            filename=f"{year} Question {i} Summary",
            title=question,
            x_axis_label="School Group",
            x_data_labels=x_labels,
            proportions=proportions_for_question,
            data_keys=groups,
            savefig=savefig,
        )
        plts.append(plt)
    return plts


def combine_years(years: list) -> pd.DataFrame:
    """
    Load and combine processed data for multiple years.
    Adds a 'Year' column to each year's DataFrame.
    """
    combined_dfs = []
    for year in years:
        df_year = load_flattened(str(year))
        df_year["Year"] = year
        combined_dfs.append(df_year)
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    return combined_df


def plot_question_trends_across_years(years: list, weight_by_parents: bool = False, savefig: bool = False):
    """
    For each non–free-response question, create a stacked bar chart showing how the whole school
    response proportions change over the specified years.

    This function combines data for the given years, then for each question computes the percentages
    of responses (e.g., Very, Satisfied, Somewhat, Not) for each year. It then calls the existing
    create_stacked_bar_chart function, using years as the x-axis labels.

    Parameters:
      years: List of years (e.g., [2023, 2024, 2025])
      weight_by_parents: Whether to weight responses by parent count.
      savefig: If True, saves each figure in the artifacts folder.
    """
    combined_df = combine_years(years)
    plts = []
    # Prepare x-axis labels from the years.

    reversed_years = list(reversed(years))

    # Loop over each non-free-response question.
    for i, question in enumerate(config.questions_for_each_school_level):
        if question in config.has_free_response:
            continue  # Skip free-response questions

        allowed_responses = config.question_responses.get(question, [])
        proportions_by_year = {}
        score_by_year = {}  # To hold the computed score for each year

        # Process each year.
        for year in years:
            # Filter the combined data for this year.
            df_year = combined_df[combined_df["Year"] == year]
            # Calculate totals for this year using the existing function.
            totals_df = calculate_question_totals(df_year, weight_by_parents)
            percentages = []
            for response in allowed_responses:
                # Find the row for the given question and response.
                row = totals_df[(totals_df["Question"] == question) & (totals_df["Response"] == response)]
                if not row.empty:
                    percentages.append(row.iloc[0].get("%_total", 0))
                else:
                    percentages.append(0)
            proportions_by_year[year] = percentages

            # Calculate the score for this year using the new ranking rule:
            # rank_value = 4 - response_index.
            score = sum((percentage / 100) * (4 - j) for j, percentage in enumerate(percentages))
            score_by_year[year] = score

        # Create x-axis labels that include the year and its computed score.
        x_labels = [f"{year}\n({score_by_year[year]:.2f})" for year in reversed_years]

        # Now call the stacked bar chart function using years as the data_keys and x-axis labels.
        plt_obj = create_stacked_bar_chart(
            filename=f"Trend Question {i} Summary",
            title=textwrap.fill(question, 60) + " (Trend Over Years)",
            x_axis_label="Year",
            x_data_labels=x_labels,
            proportions={year: proportions_by_year[year] for year in reversed_years},
            data_keys=reversed_years,  # Reversed order for display.
            savefig=savefig,
        )
        plts.append(plt_obj)

    return plts


def compute_rank_averages(row):
    """
    Given a flattened data row, compute the average rank values for:
      - Overall (across all school levels)
      - Grammar
      - Middle
      - High

    For each non–free-response question in config.questions_for_each_school_level:
      - For each school level (Grammar, Middle, High), if there is a response,
        look it up in config.question_responses[question]. The rank is computed as:
            rank_value = 4 - (index in allowed responses)
      - Sum these values separately for each school level and overall.
      - Compute the average if at least one valid response is found.

    Returns:
      A dictionary with keys:
         "Overall Average", "Grammar Average", "Middle Average", "High Average"
    """
    # Initialize accumulators for sums and counts.
    overall_sum, overall_count = 0, 0
    grammar_sum, grammar_count = 0, 0
    middle_sum, middle_count = 0, 0
    high_sum, high_count = 0, 0

    # Loop over each question defined in the configuration.
    for question in config.questions_for_each_school_level:
        # Skip free-response questions.
        if question in config.has_free_response:
            continue

        # Get the allowed responses (the rank ordering) for this question.
        allowed_responses = config.question_responses.get(question, [])

        # For each school level (Grammar, Middle, High).
        for level in config.levels:
            header = f"({level}) {question}"
            idx = config.index_map.get(header)
            if idx is None:
                continue

            cell = row[idx]
            # Process the cell only if it has a valid response.
            if cell and cell != "-":
                try:
                    # Determine the index of the cell value in the allowed responses.
                    response_index = allowed_responses.index(cell)
                except ValueError:
                    # If the response doesn't match an allowed value, skip it.
                    continue

                # Since the responses are inversely ordered, compute the rank.
                rank_value = 4 - response_index

                # Update the overall accumulators.
                overall_sum += rank_value
                overall_count += 1

                # Update the accumulators for the specific school level.
                if level == "Grammar":
                    grammar_sum += rank_value
                    grammar_count += 1
                elif level == "Middle":
                    middle_sum += rank_value
                    middle_count += 1
                elif level == "High":
                    high_sum += rank_value
                    high_count += 1

    # Calculate averages if at least one response was recorded.
    overall_avg = overall_sum / overall_count if overall_count > 0 else None
    grammar_avg = grammar_sum / grammar_count if grammar_count > 0 else None
    middle_avg = middle_sum / middle_count if middle_count > 0 else None
    high_avg = high_sum / high_count if high_count > 0 else None

    return {
        "Overall Average": overall_avg,
        "Grammar Average": grammar_avg,
        "Middle Average": middle_avg,
        "High Average": high_avg,
    }


if __name__ == "__main__":
    weight_by_parents = False
    year = "2025"
    raw_to_processed(year)
    df = load_flattened(year)
    rolled_up_data = calculate_question_totals(df, weight_by_parents)
    plot_individual_question_stacked_bars(year, rolled_up_data, True)
