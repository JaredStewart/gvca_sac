import os
import tempfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()
client = OpenAI()


class FileSplit(BaseModel):
    filename: str
    functions: List[str]


class AllFileSplits(BaseModel):
    splits: List[FileSplit]


def read_file_content(file_path: str) -> str:
    """Reads the content of the given file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_file_splits(file_content: str) -> AllFileSplits:
    """Sends the file content to the LLM to determine the optimal function splits."""
    system_prompt = """
    You are a code organization assistant. Your job is to analyze a given Python script and determine the best way to split its functions into multiple files. 
    Try to keep related functions together and ensure each file remains under 120 lines if possible. Maintain logical coherence.
    """

    file_split_prompt = f"""
    Given the following Python program:
    ```python
    {file_content}
    ```
    Suggest an optimal way to split this program into multiple files, following the given constraints.
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": file_split_prompt},
        ],
        response_format=AllFileSplits,
    )

    return completion.choices[0].message.parsed


def extract_functions(file_content: str) -> dict:
    """Extracts individual function definitions from the file content."""
    functions = {}
    lines = file_content.split("\n")
    current_func = None
    func_body = []

    for line in lines:
        if line.strip().startswith("def "):
            if current_func:
                functions[current_func] = "\n".join(func_body)
            current_func = line.split("(")[0].replace("def ", "").strip()
            func_body = [line]
        elif current_func:
            func_body.append(line)

    if current_func:
        functions[current_func] = "\n".join(func_body)

    return functions


def write_split_files(temp_dir: Path, file_splits: AllFileSplits, functions: dict):
    """Writes the split functions into separate files in the temp directory."""
    for split in file_splits.splits:
        file_path = temp_dir / split.filename
        with open(file_path, "w", encoding="utf-8") as f:
            for func in split.functions:
                if func in functions:
                    f.write(functions[func] + "\n\n")


def main(input_file: str):
    """Main function to handle file processing and splitting."""
    file_content = read_file_content(input_file)
    functions = extract_functions(file_content)
    file_splits = get_file_splits(file_content)

    temp_dir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    print(f"Created temporary directory: {temp_dir}")

    write_split_files(temp_dir, file_splits, functions)
    print(f"Split files written to: {temp_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a Python file into logically grouped files.")
    parser.add_argument("input_file", type=str, help="Path to the Python file to split")
    args = parser.parse_args()
    main(args.input_file)
