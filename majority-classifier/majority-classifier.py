import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here


    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes, counts = np.unique(y_train, return_counts=True)

    majority_class = classes[np.argmax(counts)]

    predictions = np.full(len(X_test), majority_class, dtype=int)

    return predictions
    

    