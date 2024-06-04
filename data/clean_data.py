"""
This module contains functions to clean the data.
"""
import re
from typing import List

def clean_str(string: str) -> set:
    """
    Cleans the input string by removing special characters, punctuation, and extra spaces.
    
    Args:
        string (str): The input string to be cleaned.
    
    Returns:
        set: A set of cleaned words from the input string.
    """
    string = re.sub(r"[^A-Za-z0-9!?]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return set(string.strip().lower().split())

def clean_emails(emails: List[str]) -> List[str]:
    """
    Cleans a list of emails by applying the clean_str function to each email.

    Args:
        emails (list): A list of email strings.

    Returns:
        list: A list of cleaned email strings.
    """
    return [clean_str(email) for email in emails]

