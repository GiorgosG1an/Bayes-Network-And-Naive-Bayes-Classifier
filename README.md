# Artificial Intelligence - Second Project

This project was developed for the Artificial Intelligence course at the University of Peloponnese, Department of Informatics and Telecommunications. It was created by Giannopoulos Georgios and Giannopoulos Ioannis.

## Project Overview

This project consists of two main parts:
1.  **Bayesian Network:** An implementation of a simple Bayesian network to calculate conditional probabilities.
2.  **Naive Bayes Classifier:** A classifier to distinguish between spam and ham (non-spam) emails.

### Bayesian Network

The Bayesian network is constructed with a predefined set of nodes and conditional probabilities. It is used to answer probabilistic queries, such as:
- The probability of a transaction being fraudulent.
- The probability of a transaction being fraudulent given certain evidence.

The network is defined in `main.py` and the core logic for the Bayesian network is implemented in the `bayes_networks` directory.

### Naive Bayes Classifier for Spam Detection

The second part of the project is a Naive Bayes Classifier for spam detection. The classifier is trained on the Enron email dataset. The process includes:
- **Data Handling:** Downloading and extracting the Enron dataset.
- **Data Cleaning:** Preprocessing the emails to remove noise and special characters.
- **Data Splitting:** Splitting the dataset into training and testing sets.
- **Training:** Training the Naive Bayes Classifier on the training data.
- **Prediction:** Predicting whether emails in the test set are spam or ham.
- **Evaluation:** Calculating the accuracy of the classifier.

The classifier also implements techniques to improve its performance:
- **Laplace Smoothing:** To handle cases where a word in the test set has not been seen in the training set.
- **Underflow Prevention:** To avoid numerical underflow when calculating probabilities with very small numbers.

## How to Run

To run the project, execute the following command in your terminal:

```
python main.py
```
