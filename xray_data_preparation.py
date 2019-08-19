import os
import cv2
import random
import pickle
import numpy as np

root = "./chest_xray"

testing_data = []
training_data = []
validation_data = []

for ttv in os.listdir(root):                                                    # ttv = train, test, val folders
    data_set_dir = os.path.join(root, ttv)
    data_set_img_arrays = []                                                    # data set image array plus class
    for img_class in os.listdir(data_set_dir):
        img_path = os.path.join(data_set_dir,img_class)
        img_class = 0 if img_class == "NORMAL" else 1
        for img in os.listdir(img_path):
            img_array = cv2.imread(os.path.join(img_path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(100,100))
            data_set_img_arrays.append([new_array, img_class])
            print("image: {} complete".format(img))
    np.random.shuffle(data_set_img_arrays)
    if ttv == "test":
        testing_data.append(data_set_img_arrays)
    elif ttv == "train":
        training_data.append(data_set_img_arrays)
    elif ttv == "val":
        validation_data.append(data_set_img_arrays)

with open("./testing_data.pickle", 'wb') as file:
    pickle.dump(testing_data, file)
with open("./training_data.pickle", 'wb') as file:
    pickle.dump(training_data, file)
with open("./validation_data.pickle", 'wb') as file:
    pickle.dump(validation_data, file)
