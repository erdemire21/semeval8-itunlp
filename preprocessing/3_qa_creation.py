import os
import pandas as pd

qa_csv_path = "test_qa.csv"
qa_df = pd.read_csv(qa_csv_path)

# Right now the qa_df has the following columns
# question and dataset. Convert to a json where each entry is a qa pair
qa_json = qa_df.to_dict(orient="records")

# Save the qa_json to a file
import json

qa_json_path = "all_qa.json"
with open(qa_json_path, "w") as f:
    json.dump(qa_json, f)

