import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def serialize_value(value):
    """
    Serialize a value for consistent representation.
    Converts NumPy arrays to lists and serializes using JSON for complex types.
    """
    if isinstance(value, np.ndarray):
        # Convert NumPy array to a Python list
        value = value.tolist()
    elif isinstance(value, list):
        # Ensure lists are JSON-serializable
        pass
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
        # Get the value type of the column
        value_type = df[column].dtype

        # Handle unique values, converting to JSON-compatible format if needed
        unique_values = df[column].dropna().map(serialize_value).unique()

        # Limit to 5 unique values
        limited_values = unique_values[:5]

        # Truncate long strings
        processed_values = []
        cumulative_char_count = 0

        for value in limited_values:
            # If cumulative character count exceeds 50, stop
            if cumulative_char_count > 50:
                break

            # Truncate strings longer than 100 characters
            if len(value) > 100:
                value = value[:97] + "..."

            processed_values.append(value)
            cumulative_char_count += len(value)

        # Format the line for this column
        example_values = ", ".join(processed_values)
        total_unique = len(unique_values)
        line = (f"Column Name: {column}, Data type -- {value_type}, -- Example values: {example_values},"
                f" Total unique elements: {total_unique}")

        summary_lines.append(line)


    # Combine all lines into a single string
    return intro + "\n".join(summary_lines)


parquet_directory = 'all_datasets'

# read each parquet file in the directory
files = os.listdir(parquet_directory)
print(files)
schemas = {}

for file in tqdm(files):
    if file.endswith('.parquet'):
        df = pd.read_parquet(os.path.join(parquet_directory, file))
        summary_string = get_column_unique_values_summary_string(df)
        # print(summary_string)
        file_name = file.split('.')[0]
        schemas[file_name] = summary_string

# Save the schemas to a json file utf-8 encoded and ensure ascii is false
with open('pandas_schemas.json', 'w', encoding='utf-8') as f:
    json.dump(schemas, f, ensure_ascii=False)
