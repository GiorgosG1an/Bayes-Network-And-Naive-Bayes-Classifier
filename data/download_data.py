"""
Module to download and extract email dataset.

This module contains functions to download a dataset from a given URL and extract the dataset from a tar file.
The dataset is expected to be a collection of emails, each of which is labeled as either 'ham' or 'spam'.
The emails and their corresponding labels are returned as two separate lists.
"""
import urllib.request
import tarfile

def download_dataset(url: str, filename: str):
    """
    This function downloads a dataset from the given URL and saves it to a file with the given filename.

    Args:
    - url (str): The URL of the dataset to download.
    - filename (str): The name of the file to save the downloaded dataset to.
    """
    urllib.request.urlretrieve(url, filename)

def extract_dataset(tar_filename: str):
    """
    This function extracts a dataset from a tar file.
    The dataset is expected to be a collection of emails, each of which is labeled as either 'ham' or 'spam'.
    The emails and their corresponding labels are returned as two separate lists.

    Parameters:
    - tar_filename (str): The name of the tar file to extract the dataset from.

    Returns:
    - emails (list): A list of emails extracted from the dataset. Each email is represented as a string.
    - y (list): A list of labels corresponding to the emails. Each label is either 'ham' or 'spam'.
    """
    emails = []
    y = []
    with tarfile.open(tar_filename, "r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                content = f.read()
                if 'enron1/ham' in member.name:
                    emails.append(content.decode('utf-8', errors='ignore'))
                    y.append('ham')
                elif 'enron1/spam' in member.name:
                    emails.append(content.decode('utf-8', errors='ignore'))
                    y.append('spam')
    return emails, y

