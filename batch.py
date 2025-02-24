import json
import pathlib
import io
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from tqdm import tqdm
from agents import get_pandas_code
import os
import ast
import pandas as pd
# pip install pyarrow

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the directory of the current file
os.chdir(current_file_directory)


def load_schemas(schema_path):
    """Load the pandas schemas from file."""
    with open(schema_path, encoding='utf-8') as f:
        return json.load(f)


def load_questions(qa_path):
    """Load the questions from file."""
    with open(qa_path, encoding='utf-8') as f:
        return json.load(f)#[:10]


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


def modify_parquet_paths(code, fixed_path="../datasets/", is_sample=False):
    """Modifies pd.read_parquet paths in the code to prepend a fixed path."""
    if is_sample:
        fixed_path += "sample_datasets/"
    else:
        fixed_path += "all_datasets/"
    return re.sub(
        r"pd\.read_parquet\(['\"](.*?\.parquet)['\"]\)",
        lambda match: f"pd.read_parquet('{fixed_path}{match.group(1)}')",
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
                            imported_module = __import__(module_name, fromlist=[name])
                            imports[as_name] = imported_module.__dict__[name]

            return imports
        except Exception as e:
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

        # If there's stdout output, try to evaluate it
        if output:
            try:
                eval_output = eval(output)
                if isinstance(eval_output, np.ndarray):
                    return eval_output.tolist()
                return eval_output
            except Exception:
                # If not evaluatable, return raw output
                return output
        # If no stdout, check for the last variable again (in case it's not ndarray)
        elif local_vars:
            last_var = list(local_vars.values())[-1]
            if isinstance(last_var, np.ndarray):
                return last_var.tolist()
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


def process_question_first_pass(question_data, schemas, question_index, failed_indices):
    """
    First pass: Attempt to generate valid pandas code and execute it once.
    If it fails, store the error information and the index to retry later.
    """
    try:
        MAIN_QUESTION = question_data['question']
        DATASET = question_data['dataset']
        TABLE_NAME = DATASET

        dataset_info = schemas[TABLE_NAME]

        # Generate code initially (no error_code on the first pass)
        pandas_code = get_pandas_code(DATASET, MAIN_QUESTION, dataset_info)
        question_data['pandas_code'] = pandas_code

        # Test the code on sample dataset just to see if it runs
        modified_code = modify_parquet_paths(pandas_code, is_sample=True)
        modified_code = clean_pandas_code(modified_code)
        exec_output = capture_exec_output(modified_code)

        # If there's an error string from capture_exec_output, raise an Exception to trigger retry
        if isinstance(exec_output, str) and exec_output.startswith("Error :"):
            raise Exception(exec_output)

        # If no exception, we are good. We do not mark it as failed.
        question_data['execution_error'] = False

    except Exception as e:
        # If we fail, store the error for a second pass
        question_data['execution_error'] = True
        question_data['error_msg'] = str(e)

        # We'll provide an "error_code" in the same (code, error) tuple format
        # so that get_pandas_code can try to fix it in second pass
        question_data['error_code'] = (
            clean_pandas_code(question_data.get('pandas_code', '')),
            question_data['error_msg']
        )
        # Replace the pandas_code field with the string error for clarity
        question_data['pandas_code'] = f"Error on first pass: {str(e)}"

        # Add this question to the list of failed indices
        failed_indices.append(question_index)

    return question_data


def process_question_second_pass(question_data, schemas, question_index):
    """
    Second pass: If the question was problematic, we retry by providing
    the error_code tuple (previous code, error message) to get_pandas_code.
    Then test the newly generated code again on the sample dataset.
    """
    if not question_data.get('execution_error', False):
        # If no error from first pass, just return
        return question_data

    try:
        MAIN_QUESTION = question_data['question']
        DATASET = question_data['dataset']
        TABLE_NAME = DATASET

        dataset_info = schemas[TABLE_NAME]

        # Attempt to generate a new code with the error_code from the first pass
        error_code = question_data.get('error_code', None)
        new_pandas_code = get_pandas_code(DATASET, MAIN_QUESTION, dataset_info, error_code=error_code)
        question_data['pandas_code'] = new_pandas_code

        # Test on sample dataset again
        modified_code = modify_parquet_paths(new_pandas_code, is_sample=True)
        modified_code = clean_pandas_code(modified_code)
        exec_output = capture_exec_output(modified_code)

        # If there's an error string from capture_exec_output, raise an Exception
        if isinstance(exec_output, str) and exec_output.startswith("Error :"):
            raise Exception(exec_output)

        # If we get here, the second pass succeeded
        question_data['execution_error'] = False
        question_data['error_msg'] = None
        question_data['error_code'] = None  # Clear out error info

    except Exception as e:
        # Second pass also fails. Store final error.
        question_data['execution_error'] = True
        question_data['error_msg'] = str(e)
        question_data['pandas_code'] = f"Error on second pass: {str(e)}"

    return question_data


def execute_pandas_code(data, fixed_path="../datasets/", is_sample=False):
    """
    Execute pandas code for each question and capture results.

    Args:
        data (list): A list of dictionaries containing pandas code under the 'pandas_code' key.
        fixed_path (str): The path to fix in the parquet files.
        is_sample (bool): Flag to determine whether to use sample datasets.

    Returns:
        list: The updated data with the 'final_answer' key added to each entry.
    """
    for entry in tqdm(data, desc="Executing pandas code"):
        # Extract and clean the code
        raw_code = entry.get('pandas_code', '')
        cleaned_code = clean_pandas_code(raw_code)

        # Modify the parquet paths and execute the code
        modified_code = modify_parquet_paths(cleaned_code, fixed_path=fixed_path, is_sample=is_sample)
        result = capture_exec_output(modified_code)
        entry['final_answer'] = result

    return convert_types(data)


def run_pipeline(schema_path, qa_path, output_path, sample_output_path):
    """
    Run the complete pipeline in two passes for code generation,
    then execute the code on both full and sample datasets.
    1) First pass: Generate code & attempt to run. If error, store index.
    2) Second pass: Retry only for those indices that failed.
    3) Execute final code on both full and sample datasets.
    """
    # Load input data
    schemas = load_schemas(schema_path)
    questions = load_questions(qa_path)

    # -----------------------------
    # First pass: generate code
    # -----------------------------
    print("Generating pandas code (first pass)...")
    results = []
    failed_indices = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for i, result in enumerate(
            tqdm(executor.map(
                lambda idx_q: process_question_first_pass(idx_q[1], schemas, idx_q[0], failed_indices),
                enumerate(questions)
            ), total=len(questions))
        ):
            results.append(result)

    # -----------------------------
    # Second pass: retry failures
    # -----------------------------
    if failed_indices:
        print("\nRetrying problematic questions (second pass)...")
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Map over the indices that failed
            for i, updated_result in enumerate(
                tqdm(executor.map(
                    lambda idx: process_question_second_pass(results[idx], schemas, idx),
                    failed_indices
                ), total=len(failed_indices))
            ):
                # Update the results in place
                fail_idx = failed_indices[i]
                results[fail_idx] = updated_result

    # After second pass, we have final results (some may still have errors)

    # Save intermediate results (just code, not executed against the full dataset)
    intermediate_file = "all_qa_pandas_code_not_executed.json"
    with open(intermediate_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # ----------------------------------
    # Execute code on full datasets
    # ----------------------------------
    print("\nExecuting code on full datasets...")
    full_results = execute_pandas_code(results.copy())
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=4)

    # ----------------------------------
    # Execute code on sample datasets
    # ----------------------------------
    print("\nExecuting code on sample datasets...")
    sample_results = execute_pandas_code(results.copy(), is_sample=True)
    pathlib.Path(sample_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(sample_output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Define paths
    SCHEMA_PATH = '../2_Schema_Creation/pandas_schemas.json'
    QA_PATH = '../3_qa_creation/all_qa.json'
    OUTPUT_PATH = 'intermediate_results/code_execution_results.json'
    SAMPLE_OUTPUT_PATH = 'intermediate_results/code_execution_results_sample.json'

    # Run the pipeline
    run_pipeline(SCHEMA_PATH, QA_PATH, OUTPUT_PATH, SAMPLE_OUTPUT_PATH)
