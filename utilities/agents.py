from pydantic import BaseModel
from typing import List, Union
import openai
from utilities.utils import update_counter, get_text_after_last_think_tag
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ERROR_LLM_PROVIDER = openai.OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)
MAIN_LLM_PROVIDER = openai.OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)

MAIN_LLM = os.getenv("MAIN_LLM", "deepseek-ai/DeepSeek-R1")  # Default model
ERROR_LLM = os.getenv("ERROR_LLM", "deepseek-ai/DeepSeek-R1")  # Error handling model

def get_pandas_code(dataset_name, question, schema, temperature=0, error_code=None):
    """
    Generates Python code using pandas to answer a given question based on a dataset schema.
    If error_code is provided, it attempts to fix the error in the previous code.

    Parameters:
    dataset_name (str): The name of the dataset.
    question (str): The question to be answered using the dataset.
    schema (str): The schema of the dataset.
    temperature (float): Temperature for LLM generation.
    error_code (tuple, optional): Tuple containing (previous_code, error_message).

    Returns:
    str: The generated Python code as a string.
    """
    instructions = '''The code should return a print statement with the answer to the question.
    The code should leave the answer be and not print anything other than the variable that holds the answer.
    Please write a single Python code block that answers the following question and prints the result in one line at the end.'''

    unique_keywords = ['unique', 'different', 'distinct']
    if all(keyword not in question.lower() for keyword in unique_keywords):
        instructions += '''
        If the question doesn't specifically ask for it, don't use unique() or drop_duplicates() functions.'''

    instructions += '''
    If it is a Yes or No question, the answer should be a boolean.
    Do not include any explanations, comments, or additional code blocks.
    Do not print intermediate steps just the answer.
    Do not interact with the user.
    Never display any sort of dataframes or tables.
    Your output can never take more than a single line after printing and it can never be any sort of objects such as pandas or numpy objects, series etc. 
    Your output must be one of the following:

    Boolean: True/False
    Category/String: A value
    Number: A numerical value
    List[category/string]: ['cat', 'dog']
    List[number]: [1, 2, 3]
    So the outputs have to be native python

    '''

    user_prompt = f'''Given the dataset schema {schema}
                Generate a python code to answer this question: `{question}` that strictly follows the instructions below:
                {instructions}`:'''

    user_prompt += (
        f"The following python code made for pandas for the parquet file {dataset_name}.parquet reads the parquet file and "
        f"running it returns the answer that is enough to answer the question `{question}`"
        )

    if error_code:
        prev_code, error_msg = error_code

        user_prompt = f'''
                    Please fix the code to properly answer the question: `{question}`
                    Dataset schema: {schema}
                    Follow these instructions:
                    {instructions}
                    The following code generated an error when executed:
                    ```python
                    {prev_code}
                    ```
                    Error: {error_msg} Solve the error and provide the corrected code'''
        user_prompt += (
            f"The following python code made for pandas for the parquet file {dataset_name}.parquet reads the parquet file and "
            f"running it returns the answer that is enough to answer the question `{question}` with the error fixed")

    CURRENT_LLM = ERROR_LLM if error_code else MAIN_LLM
    CURRENT_PROVIDER = ERROR_LLM_PROVIDER if error_code else MAIN_LLM_PROVIDER

    chat_completion = CURRENT_PROVIDER.chat.completions.create(
        model=CURRENT_LLM,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=5000,
    )
    to_return = get_text_after_last_think_tag(chat_completion.choices[0].message.content)
    return to_return
