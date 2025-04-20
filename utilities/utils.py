import pandas as pd
import os

def get_text_after_last_think_tag(text):
    last_think_index = text.rfind("</think>")
    if last_think_index != -1:
        # Get everything after the last </think> tag
        return get_text_after_last_think_tag(text[last_think_index + len("</think>"):])
    last_think_index = text.rfind("<think>")
    if last_think_index != -1:
        # Get everything after the last <think> tag
        return get_text_after_last_think_tag(text[last_think_index + len("<think>"):])
    # If neither tag is found, return the original text
    return text  # Return the whole text if </think> is not found