import numpy as np

def load_data():
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")

    return {"X": X_train, "y": y_train}, {"X": X_test, "y": None}