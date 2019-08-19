import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle

test_data = pickle.load(open("testing_data.pickle", "rb"))
training_data = pickle.load(open("training_data.pickle", "rb"))
validation_data = pickle.load(open("validation_data.pickle", "rb"))


def create_X_and_y(data, type=None):
    X = []
    y = []
    for features, labels in data[0]:
        X.append(features)
        y.append(labels)
    X = np.array(X)
    y = np.array(y)

    print(X.reshape(-1,100,100,1))

    return X,y

X_test, y_test = create_X_and_y(test_data, type="test")
