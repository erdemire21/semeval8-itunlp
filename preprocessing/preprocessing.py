import re
import os
import json

import pandas as pd
import numpy as np
from tqdm import tqdm

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the directory of the current file
os.chdir(current_file_directory)

def load_sample(dataset):
    """Load a sample parquet file for a given dataset."""
    sample_dataset = pd.read_parquet(f"competition/{dataset}/sample.parquet")
    return sample_dataset


def load_table(name):
    """Load the full parquet table for a given dataset."""
    return pd.read_parquet(f"competition/{name}/all.parquet")


def rename_columns_for_sql(df):
    """
    Renames DataFrame columns to be SQL-friendly:
    - Replaces spaces and special characters with underscores, except at the end where it is replaced with an empty string.
    - Converts column names to lowercase.
    - Ensures column names are unique.
    - Ensures column names start with a letter.
    
    Parameters:
    df (pd.DataFrame): The DataFrame whose columns need to be renamed.

    Returns:
    pd.DataFrame: A new DataFrame with renamed columns.
    """
    column_count = {}
    new_columns = []
    
    for col in df.columns:
        # Replace spaces and special characters with underscores except at the end
        new_col = re.sub(r'\W+(?=\w)', '_', col)
        # Replace special characters at the end with an empty string
        new_col = re.sub(r'\W+$', '', new_col)
        # Convert to lowercase
        new_col = new_col.lower()
        # Ensure column starts with a letter
        if not re.match(r'^[a-zA-Z]', new_col):
            new_col = 'col_' + new_col
        # Ensure uniqueness
        if new_col in column_count:
            column_count[new_col] += 1
            new_col = f"{new_col}_{column_count[new_col]}"
        else:
            column_count[new_col] = 1
        new_columns.append(new_col)
    
    df = df.copy()
    df.columns = new_columns
    return df


def serialize_value(value):
    """
    Serialize a value for consistent representation.
    Converts NumPy arrays to lists and serializes using JSON for complex types.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, list):
        pass  # Lists are expected to be JSON-serializable
    return json.dumps(value) if isinstance(value, (list, dict)) else str(value)


def get_column_unique_values_summary_string(df):
    """
    Generate a string summary of column names, value types, unique values,
    and total number of unique items for a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        str: A formatted string summarizing the DataFrame.
    """
    summary_lines = []
    intro = 'Here are the columns for the dataset \n'
    
    for column in df.columns:
        value_type = df[column].dtype
        unique_values = df[column].dropna().map(serialize_value).unique()
        limited_values = unique_values[:5]
        processed_values = []
        cumulative_char_count = 0
        
        for value in limited_values:
            if cumulative_char_count > 50:
                break
            if len(value) > 100:
                value = value[:97] + "..."
            processed_values.append(value)
            cumulative_char_count += len(value)
        
        example_values = ", ".join(processed_values)
        total_unique = len(unique_values)
        line = (f"Column Name: {column}, Data type -- {value_type}, -- Example values: {example_values},"
                f" Total unique elements: {total_unique}")
        summary_lines.append(line)
    
    return intro + "\n".join(summary_lines)


def main(test_qa_path, output_root, sample_datasets_dir, all_datasets_dir, schema_output_path, qa_json_output_path):
    # Step 1: Fixing and Creating Datasets 
    print("Step 1: Processing datasets and creating parquet files...")
    # Read the test_qa file to get dataset names
    df = pd.read_csv(test_qa_path)
    datasets = df['dataset'].unique()

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        df_sample = load_sample(dataset)
        df_table = load_table(dataset)

        df_sample = rename_columns_for_sql(df_sample)
        df_table = rename_columns_for_sql(df_table)

        df_sample.to_parquet(os.path.join(sample_datasets_dir, f"{dataset}.parquet"))
        df_table.to_parquet(os.path.join(all_datasets_dir, f"{dataset}.parquet"))

    # Step 2: Creating Schema Summary
    print("Step 2: Generating schema summaries for all datasets...")
    files = os.listdir(all_datasets_dir)
    print(f"Parquet files found: {files}")
    schemas = {}

    for file in tqdm(files):
        if file.endswith('.parquet'):
            file_path = os.path.join(all_datasets_dir, file)
            df_parquet = pd.read_parquet(file_path)
            summary_string = get_column_unique_values_summary_string(df_parquet)
            file_name = file.split('.')[0]
            schemas[file_name] = summary_string

    with open(schema_output_path, 'w', encoding='utf-8') as f:
        json.dump(schemas, f, ensure_ascii=False, indent=4)

    # Step 3: Creating QA JSON file from QA CSV
    print("Step 3: Creating QA JSON file...")
    qa_df = pd.read_csv(test_qa_path)
    qa_json = qa_df.to_dict(orient="records")
    with open(qa_json_output_path, "w") as f:
        json.dump(qa_json, f, indent=2)

    print("All processing complete.")


if __name__ == "__main__":
    # Path to the competition folder which contains the test_qa.csv file. It should also contain the sample and all parquet files.
    TEST_QA_PATH = 'competition/test_qa.csv'  
    
    # Output directory structure
    OUTPUT_ROOT = os.path.join("..", "data")  # Root directory for all output files
    SAMPLE_DATASETS_DIR = os.path.join(OUTPUT_ROOT, "sample_datasets")  # Directory for sample dataset parquet files
    ALL_DATASETS_DIR = os.path.join(OUTPUT_ROOT, "all_datasets")  # Directory for complete dataset parquet files
    SCHEMA_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, 'pandas_schemas.json')  # Path for the schema summary JSON
    QA_JSON_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, "all_qa.json")  # Path for the processed QA JSON file
    
    # Create output directories if they don't exist
    for directory in [OUTPUT_ROOT, SAMPLE_DATASETS_DIR, ALL_DATASETS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    main(TEST_QA_PATH, OUTPUT_ROOT, SAMPLE_DATASETS_DIR, ALL_DATASETS_DIR, SCHEMA_OUTPUT_PATH, QA_JSON_OUTPUT_PATH) 