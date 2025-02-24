import re
import pandas as pd
import os

df = pd.read_csv('competition/test_qa.csv')

# Get unique dataset names
datasets = df['dataset'].unique()

def load_sample(dataset):
    sample_dataset = pd.read_parquet(f"competition/{dataset}/sample.parquet")
    return sample_dataset

def load_table(name):
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
    # Initialize a dictionary to keep track of column name counts for uniqueness
    column_count = {}
    new_columns = []

    for col in df.columns:
        # Replace spaces and special characters with underscores, except at the end
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

    # Rename the columns in the DataFrame
    df = df.copy()
    df.columns = new_columns
    return df

for i in datasets:
    df1 = load_sample(i)
    df2 = load_table(i)

    df1 = rename_columns_for_sql(df1)
    df2 = rename_columns_for_sql(df2)

    if not os.path.exists("sample_datasets"):
        os.makedirs("sample_datasets")

    if not os.path.exists("all_datasets"):
        os.makedirs("all_datasets")

    df1.to_parquet(f"sample_datasets/{i}.parquet")
    df2.to_parquet(f"all_datasets/{i}.parquet")
