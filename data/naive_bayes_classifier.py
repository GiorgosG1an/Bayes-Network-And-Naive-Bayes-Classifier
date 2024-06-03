import numpy as np
from collections import defaultdict
from typing import List, Set

class NaiveBayesClassifier:
    def __init__(self):
        self.spam_word_probs = defaultdict(float)
        self.ham_word_probs = defaultdict(float)
        self.p_spam = 0
        self.p_ham = 0
        self.spam_word_count = defaultdict(int)
        self.ham_word_count = defaultdict(int)
        self.spam_email_count = 0
        self.ham_email_count = 0
        self.total_emails = 0

    def train(self, emails: List[Set[str]], labels: List[str]):
        self.total_emails = len(emails)
        self.spam_email_count = sum(1 for label in labels if label == 'spam')
        self.ham_email_count = sum(1 for label in labels if label == 'ham')
        
        self.p_spam = self.spam_email_count / self.total_emails
        self.p_ham = self.ham_email_count / self.total_emails

        for email, label in zip(emails, labels):
            if label == 'spam':
                for word in email:
                    self.spam_word_count[word] += 1
            elif label == 'ham':
                for word in email:
                    self.ham_word_count[word] += 1

        # Calculate probabilities
        vocab = set(self.spam_word_count.keys()).union(set(self.ham_word_count.keys()))
        vocab_size = len(vocab)
        
        for word in vocab:
            self.spam_word_probs[word] = (self.spam_word_count[word] + 1) / (self.spam_email_count + vocab_size)
            self.ham_word_probs[word] = (self.ham_word_count[word] + 1) / (self.ham_email_count + vocab_size)

    def predict(self, emails: List[Set[str]]) -> List[str]:
        y_pred = []
        for email in emails:
            p_spam_given_email = np.log(self.p_spam)
            p_ham_given_email = np.log(self.p_ham)

            for word in email:
                if word in self.spam_word_probs:
                    p_spam_given_email += np.log(self.spam_word_probs[word])
                if word in self.ham_word_probs:
                    p_ham_given_email += np.log(self.ham_word_probs[word])
                    
            if p_spam_given_email > p_ham_given_email:
                y_pred.append('spam')
            else:
                y_pred.append('ham')

        return y_pred
