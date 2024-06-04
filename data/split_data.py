"""Module to split the given emails and labels into training and testing sets."""

import numpy as np
from typing import List

def split_data(emails: List, y: List, train_ratio=0.8, seed=13):
    """
    Split the given emails and labels into training and testing sets.

    Parameters:
    - emails (list): List of email data. Each element of the list is expected to be a string representing an email.
    - labels (list): List of corresponding labels. Each element of the list is expected to be a string representing a label.
    - train_ratio (float, optional): Ratio of data to be used for training. The value should be between 0 and 1. Defaults to 0.8.
    - seed (int, optional): Seed value for random number generation. This is used to ensure that the random shuffling is reproducible. Defaults to 13.

    Returns:
    - emails_train (list): List of emails for training.
    - labels_train (list): List of labels for training.
    - emails_test (list): List of emails for testing.
    - labels_test (list): List of labels for testing.
    """
    np.random.seed(seed)
    N = len(emails)
    idx = np.random.permutation(N)
    
    emails_train = [emails[idx[i]] for i in range(int(0.8*N))]
    y_train = [y[idx[i]] for i in range(int(0.8*N))]

    emails_test = [emails[idx[i]] for i in range(int(0.8*N), N)]
    y_test = [y[idx[i]] for i in range(int(0.8*N), N)]
    
    return emails_train, y_train, emails_test, y_test
