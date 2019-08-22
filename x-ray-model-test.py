import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Accuracy
import numpy as np
import pickle
import time
from sklearn.metrics import accuracy_score

test_data = pickle.load(open("testing_data.pickle", "rb"))
def create_X_and_y(data):
    X = []
    y = []
    for features, labels in data[0]:
        X.append(features)
        y.append(labels)
    X = np.array(X).reshape(-1,100,100,1)                                       # len dataset(600) // pixels_dim1(100) // pixels_dim2(100) // colour_numbers(1)
    y = np.array(y)
    return X,y
X_test, y_test = create_X_and_y(test_data)
y_test = y_test.reshape(-1,1)

model = tf.keras.models.load_model("64x2-CONV-NET.model")
predict = model.predict([X_test])
predict = predict.astype(int)

print(accuracy_score(predict, y_test))
# only 75.4% accuracy; however,skipped fine tuning due to data set size.
