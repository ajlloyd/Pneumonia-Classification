import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time

NAME = "Pneumonia-vs-unaffected-64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir=".\logs\\".format(NAME))

test_data = pickle.load(open("testing_data.pickle", "rb"))
training_data = pickle.load(open("training_data.pickle", "rb"))
validation_data = pickle.load(open("validation_data.pickle", "rb"))

def create_X_and_y(data):
    X = []
    y = []
    for xs, ys in data[0]:
        X.append(xs)
        y.append(ys)
    X = np.array(X).reshape(-1,100,100,1)                                       # len dataset(600) // pixels_dim1(100) // pixels_dim2(100) // colour_numbers(1)
    y = np.array(y)
    return X,y

X_train, y_train = create_X_and_y(training_data)
X_train = X_train / 255

model = Sequential()
# Layer 1:
model.add(Conv2D(64, (3,3), padding="same", activation="relu", input_shape = X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
# Layer 2:
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
# Layer 3:
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=30, epochs=5, validation_split=0.3, callbacks=[tensorboard])

model.save("64x2-CONV-NET.model")
