import numpy as np

def split_data(emails, labels, train_ratio=0.8, seed=13):
    np.random.seed(seed)
    N = len(emails)
    idx = np.random.permutation(N)
    split_point = int(train_ratio * N)
    
    emails_train = [emails[i] for i in idx[:split_point]]
    labels_train = [labels[i] for i in idx[:split_point]]
    
    emails_test = [emails[i] for i in idx[split_point:]]
    labels_test = [labels[i] for i in idx[split_point:]]
    
    return emails_train, labels_train, emails_test, labels_test
