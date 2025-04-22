import json
import pathlib
import io
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from tqdm import tqdm
from utilities.agents import get_pandas_code
import os
import ast
import pandas as pd

import logging
import traceback

# Setup error logging
MAIN_LLM_ENV = os.environ.get("MAIN_LLM", "default")
# sanitize MAIN_LLM so that any slashes become underscores
sanitized_name = re.sub(r"[\\/]", "_", MAIN_LLM_ENV)
logs_dir = "error_logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"{sanitized_name}.log")

logger = logging.getLogger("error_logger")
handler = logging.FileHandler(log_file)
formatter = logging.Formatter("%(asctime)s %(threadName)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def classify_error(exc):
    """Classify an exception into syntax, logic, data-type, or other."""
    msg = str(exc).lower()
    if isinstance(exc, SyntaxError) or "syntax error" in msg or "unexpected eof" in msg:
        return "syntax"
    elif isinstance(exc, TypeError) or "dtype" in msg or "typeerror" in msg:
        return "data-type"
    elif "groupby" in msg or "aggregation" in msg or "cannot insert" in msg:
        return "logic"
    else:
        return "other"


def load_schemas(schema_path):
    """Load the pandas schemas from file."""
    with open(schema_path, encoding='utf-8') as f:
        return json.load(f)


def load_questions(qa_path):
    """Load the questions from file."""
    with open(qa_path, encoding='utf-8') as f:
        return json.load(f)


def process_question(question_data, schemas, dataset_folder_path, max_retries=1):
    """Process a single question to generate pandas code with error checking and retrying."""
    # initialize per-question error history
    question_data.setdefault("error_history", [])

    # NEW -------------  keep track of *all* failed code/error pairs -------------
    previous_attempts = []          # [(code, error_msg), ...]
    # ---------------------------------------------------------------------------

    try:
        MAIN_QUESTION = question_data['question']
        DATASET = question_data['dataset']
        TABLE_NAME = DATASET

        dataset_info = schemas[TABLE_NAME]
        error_code = None
        pandas_code = get_pandas_code(DATASET, MAIN_QUESTION, dataset_info)

        # Save original code before path modification
        original_code = clean_pandas_code(pandas_code)
        
        # Test the code on sample dataset
        modified_code = modify_parquet_paths(pandas_code, dataset_folder_path=dataset_folder_path, is_sample=True)
        modified_code = clean_pandas_code(modified_code)
        retries = 0

        exec_output = ""

        while retries <= max_retries:
            try:
                # Try executing the code
                exec_output = capture_exec_output(clean_pandas_code(modified_code))
                if isinstance(exec_output, str) and 'Error' in exec_output:
                    raise Exception(exec_output)

                # successful execution
                question_data["status"] = "success"
                break  # If successful, break the loop

            except Exception as exec_error:
                # classify the error
                category = classify_error(exec_error)
                tb = traceback.format_exc()

                # append to this question's history, including the code that caused the error
                question_data["error_history"].append({
                    "iteration": retries,
                    "error_type": category,
                    "exception": type(exec_error).__name__,
                    "message": str(exec_error),
                    "traceback": tb,
                    "code": modified_code
                })

                # log to the shared error log
                logger.error(
                    "Question: %s | Iteration: %d | Type: %s | Exception: %s | Message: %s | Code: %s",
                    MAIN_QUESTION, retries, category, type(exec_error).__name__, str(exec_error), modified_code
                )

                # ------------ keep a concise record for the next LLM call ---------
                # Use original code (without path modifications) for error reporting to LLM
                previous_attempts.append(
                    (original_code, str(exec_error))
                )
                if retries == max_retries:
                    break

                # If there's an error and we have retries left, try to fix it
                if len(previous_attempts) == 1:
                    error_arg = previous_attempts[0]           # tuple, old behaviour
                else:
                    error_arg = previous_attempts[:]           # list – new behaviour

                pandas_code = get_pandas_code(
                    DATASET,
                    MAIN_QUESTION,
                    dataset_info,
                    error_code=error_arg
                )
                
                # Update original code with the new code from LLM
                original_code = clean_pandas_code(pandas_code)
                
                modified_code = modify_parquet_paths(
                    pandas_code,
                    dataset_folder_path=dataset_folder_path,
                    is_sample=True
                )
                modified_code = clean_pandas_code(modified_code)
                retries += 1

        # if we never succeeded, mark as failed
        if question_data.get("status") != "success":
            question_data["status"] = "failed"

        question_data['pandas_code'] = pandas_code
        return question_data

    except Exception as e:
        # catch any unexpected top-level error
        question_data["status"] = "failed"
        category = classify_error(e)
        tb = traceback.format_exc()
        # include whatever pandas_code was last set (if any)
        last_code = question_data.get('pandas_code', '')
        question_data["error_history"].append({
            "iteration": None,
            "error_type": category,
            "exception": type(e).__name__,
            "message": str(e),
            "traceback": tb,
            "code": last_code
        })
        logger.error(
            "Question: %s | Top-level Exception: %s | Message: %s | Code: %s",
            question_data.get('question'), type(e).__name__, str(e), last_code
        )
        question_data['pandas_code'] = str(e)
        return question_data


def modify_parquet_paths(code, dataset_folder_path="../datasets/", is_sample=False):
    """Modifies pd.read_parquet paths in the code to prepend a fixed path."""
    if is_sample:
        dataset_folder_path += "sample_datasets/"
    else:
        dataset_folder_path += "all_datasets/"
    return re.sub(
        r"pd\.read_parquet\(['\"](.*?\.parquet)['\"]\)",
        lambda match: f"pd.read_parquet('{dataset_folder_path}{match.group(1)}')",
        code
    )


def capture_exec_output(code):
    """
    Execute code and return its output in its original format. If no output,
    return 'None'. If an error occurs, return the exception.

    Dynamically extracts imports from the code and includes them in the execution context.
    """

    def extract_imports(code):
        """
        Extract all imported modules and objects from the given code.
        Returns a dictionary of imported modules and their names.
        If an error occurs during parsing, returns an empty dictionary.
        """
        try:
            tree = ast.parse(code)
            imports = {}

            for node in ast.walk(tree):
                # Handle `import` statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        as_name = alias.asname or alias.name
                        imports[as_name] = __import__(module_name)

                # Handle `from ... import ...` statements
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    for alias in node.names:
                        name = alias.name
                        as_name = alias.asname or alias.name
                        if module_name:
                            full_name = f"{module_name}.{name}"
                            imports[as_name] = __import__(module_name, fromlist=[name]).__dict__[name]

            return imports
        except Exception:
            # Return empty dictionary if any error occurs during import extraction
            return {}

    # Extract imports from the code
    dynamic_imports = extract_imports(code)

    # Prepare the execution environment with built-ins and dynamic imports
    execution_globals = {"__builtins__": __builtins__, "np": np, "pd": pd, "ast": ast}
    execution_globals.update(dynamic_imports)

    f = io.StringIO()
    try:
        local_vars = {}
        with redirect_stdout(f):
            exec(code, execution_globals, local_vars)

        # Check if there are any local variables
        if local_vars:
            # Get the last defined variable
            last_var = list(local_vars.values())[-1]
            if isinstance(last_var, np.ndarray):
                return last_var.tolist()  # Convert NumPy array to Python list

        # If last variable is not a NumPy ndarray, proceed to capture stdout
        output = f.getvalue().strip()

        # If there's stdout output, return it
        if output:
            try:
                eval_output = eval(output)
                if isinstance(eval_output, np.ndarray):
                    return eval_output.tolist()  # Convert NumPy array to Python list
                return eval_output
            except Exception:
                return output  # If not evaluatable, return raw output
        # If no stdout, check for the last variable again (in case it's not ndarray)
        elif local_vars:
            last_var = list(local_vars.values())[-1]
            if isinstance(last_var, np.ndarray):
                return last_var.tolist()  # Convert NumPy array to Python list
            return last_var
        else:
            return 'None'  # No output, no variables
    except Exception as e:
        return "Error :" + str(e)  # Return exception as a string


def convert_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_types(value) for key, value in obj.items()}
    else:
        return obj


def clean_pandas_code(raw_code):
    """
    Clean and extract Python code from a raw string.

    Args:
        raw_code (str): The raw string containing Python code with possible markdown formatting.

    Returns:
        str: The cleaned Python code.
    """
    raw_code = raw_code.strip()
    if '```python' in raw_code:
        # Extract everything between '```python' and the next '```'
        cleaned_code = raw_code.split('```python', 1)[1].split('```', 1)[0].strip()
    else:
        # Otherwise, get everything up to the first ```
        cleaned_code = raw_code.split('```', 1)[0].strip()
    return cleaned_code


def execute_pandas_code(data, dataset_folder_path="../datasets/", is_sample=False):
    """
    Execute pandas code for each question and capture results.

    Args:
        data (list): A list of dictionaries containing pandas code under the 'pandas_code' key.
        dataset_folder_path (str): The path to fix in the parquet files.
        is_sample (bool): Flag to determine whether to use sample datasets.

    Returns:
        list: The updated data with the 'final_answer' key added to each entry.
    """
    for entry in tqdm(data, desc="Executing pandas code"):
        # Extract and clean the code
        raw_code = entry.get('pandas_code', '')
        cleaned_code = clean_pandas_code(raw_code)

        # Modify the parquet paths and execute the code
        modified_code = modify_parquet_paths(cleaned_code, dataset_folder_path=dataset_folder_path, is_sample=is_sample)
        result = capture_exec_output(modified_code)
        entry['final_answer'] = result

    return convert_types(data)


def run_pipeline(schema_path, qa_path, output_path, sample_output_path, max_retries=1, dataset_folder_path="data/"):
    """Run the complete pipeline with error checking and retrying."""
    # Load input data
    schemas = load_schemas(schema_path)
    questions = load_questions(qa_path)

    # Generate pandas code with error checking
    print("Generating pandas code with error checking...")
    results = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        for result in tqdm(executor.map(lambda q: process_question(q, schemas, dataset_folder_path, max_retries), questions), total=len(questions)):
            results.append(result)

    # Save intermediate results
    intermediate_file = "intermediate_results/all_qa_pandas_code_not_executed.json"
    pathlib.Path(intermediate_file).parent.mkdir(parents=True, exist_ok=True)
    with open(intermediate_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Execute code and save results for both full and sample datasets
    print("Executing code on full datasets...")
    full_results = execute_pandas_code(results.copy(), dataset_folder_path=dataset_folder_path)
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=4)

    print("Executing code on sample datasets...")
    sample_results = execute_pandas_code(results.copy(), dataset_folder_path=dataset_folder_path, is_sample=True)
    pathlib.Path(sample_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(sample_output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Define paths
    SCHEMA_PATH = 'data/pandas_schemas.json'
    QA_PATH = 'data/all_qa.json'
    OUTPUT_PATH = 'intermediate_results/code_execution_results.json'
    SAMPLE_OUTPUT_PATH = 'intermediate_results/code_execution_results_sample.json'
    DATASET_FOLDER_PATH = 'data/'

    # Run the pipeline with 1 retry attempt and default dataset folder path
    run_pipeline(SCHEMA_PATH, QA_PATH, OUTPUT_PATH, SAMPLE_OUTPUT_PATH, max_retries=2, dataset_folder_path=DATASET_FOLDER_PATH)

    # Just log the location without displaying contents
    print(f"\nError log saved to: {log_file}")
