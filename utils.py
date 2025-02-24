import pandas as pd
import os

def update_counter():
    """
    Updates the counter in predictions/counter.txt if attempt > 0.
    Creates the file and initializes the counter if it does not exist.

    Parameters:
    attempt (int): The attempt value to check.
    """
    # Path to the counter file
    counter_file_path = "predictions/counter.txt"

    # Ensure the predictions directory exists
    os.makedirs("predictions", exist_ok=True)

    if True:
        # Check if the counter file exists
        if not os.path.exists(counter_file_path):
            # If not, create it and initialize the counter
            with open(counter_file_path, "w") as counter_file:
                counter_file.write("1")
            counter = 1
        else:
            # If it exists, read the current count, increment it, and write it back
            with open(counter_file_path, "r+") as counter_file:
                counter = int(counter_file.read())
                counter += 1
                counter_file.seek(0)
                counter_file.write(str(counter))
                counter_file.truncate()


def get_text_after_last_think_tag(text):
    last_think_index = text.rfind("</think>")
    if last_think_index != -1:
        # Get everything after the last </think> tag
        return text[last_think_index + len("</think>"):]
    return text  # Return the whole text if </think> is not found