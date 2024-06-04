"""
## Naive Bayes Classifier

This module contains the Naive Bayes Classifier for spam detection. 

Classes:
- NaiveBayesClassifier: Naive Bayes Classifier for spam detection.

The NaiveBayesClassifier class has methods for: 
- initializing the classifier
- training the classifier
- predicting the labels of the given emails.

University: University of Peloponnese, Department of Informatics and Telecommunications

Course: Artificial Intelligence

Authors: 
- Giannopoulos Georgios
- Giannopoulos Ioannis
"""
from collections import defaultdict
from typing import List, Set

class NaiveBayesClassifier():
    """
    Naive Bayes Classifier for spam detection.
    """
    
    def __init__(self) -> None:
        """
        Initialize the Naive Bayes Classifier.
        """
        self.spam_word_count = defaultdict(int)
        """Number of times a word appears in spam emails"""
        self.ham_word_count = defaultdict(int)
        """Number of times a word appears in ham emails"""
        self.spam_email_count = 0
        """Number of spam emails"""
        self.ham_email_count = 0
        """Number of ham emails"""
        self.total_emails = 0
        """Total number of emails"""
        self.p_spam = 0
        """Probability of spam emails"""
        self.p_ham = 0
        """Probability of ham emails"""
        self.p_word_given_spam = defaultdict(float)
        """Probability of a word given that the email is spam"""
        self.p_word_given_ham = defaultdict(float)
        """Probability of a word given that the email is ham"""
        self.words = set()
        """Set of all unique words in spam and ham emails"""

    def train(self, emails, labels):
        """
        Train the Naive Bayes Classifier.

        Calculate the probabilities of spam and ham emails and 
        the probability of each word given that the email is spam or ham.
        """
        self.total_emails = len(emails)
        self.spam_email_count = sum(1 for label in labels if label == 'spam')
        self.ham_email_count = sum(1 for label in labels if label == 'ham')

        # Calculate the probability of spam and ham emails
        self.p_spam = self.spam_email_count / self.total_emails
        self.p_ham = self.ham_email_count / self.total_emails

        # Count the number of times each word appears in spam and ham emails
        for email, label in zip(emails, labels):
            if label == 'spam':
                for word in email:
                    self.spam_word_count[word] += 1
            elif label == 'ham':
                for word in email:
                    self.ham_word_count[word] += 1

        # Create a set that will contain all unique words from spam and ham emails
        # so we can iterate all the words to calculate the probabilities
        self.words = set(self.spam_word_count.keys()).union(set(self.ham_word_count.keys()))

        # Calculate the probability of each word given that the email is spam or ham
        for word in self.words:
            self.p_word_given_spam[word] = self.spam_word_count[word] / self.spam_email_count
            self.p_word_given_ham[word] = self.ham_word_count[word] / self.ham_email_count
        
    def predict(self, emails) -> List[str]:
        """
        Predict the labels of the given emails.
        """
        y_pred = []

        # iterate over each email
        for email in emails:
            p_spam_given_email = self.p_spam
            p_ham_given_email = self.p_ham

            # iterate over each word in the email
            for word in email:

                # if the word is in the set of unique words, update the probabilities, otherwise the word was not in the training set so we ignore it
                if word in self.words:
                    p_spam_given_email *= self.p_word_given_spam[word]
                    p_ham_given_email *= self.p_word_given_ham[word]

            # predict the label based on the probabilities
            if p_spam_given_email > p_ham_given_email:
                y_pred.append('spam')
            else:
                y_pred.append('ham')
        
        return y_pred
