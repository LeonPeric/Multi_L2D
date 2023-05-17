import numpy as np
from PIL import Image
import os
from torchvision import transforms
import pickle

folder_train = "data/train/"
folder_benign_train = 'data/train/benign/'
folder_malignant_train = 'data/train/malignant/'

folder_test = "data/test/"
folder_benign_test = 'data/test/benign/'
folder_malignant_test = 'data/test/malignant/'

num_samples_train = 2637
num_samples_test = 660

X_train = np.zeros((num_samples_train, 3, 224, 224), dtype='float')
Y_train = np.zeros(num_samples_train, dtype=int)

preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

image_count = 0

for dir in os.listdir(folder_train):
    if dir == "benign":
        label = 0
    else:
        label = 1
    for filename in os.listdir(folder_train + dir):
        input_image = Image.open(folder_train + dir + "/" + filename)
        img = preprocess(input_image)
        X_train[image_count] = img
        Y_train[image_count] = label
        image_count += 1


X_test = np.zeros((num_samples_test, 3, 224, 224), dtype='float')
Y_test = np.zeros(num_samples_test, dtype=int)

image_count = 0
for dir in os.listdir(folder_test):
    if dir == "benign":
        label = 0
    else:
        label = 1
    for filename in os.listdir(folder_test + dir):
        input_image = Image.open(folder_test + dir + "/" + filename)
        img = preprocess(input_image)
        X_test[image_count] = img
        Y_test[image_count] = label
        image_count += 1

data = {}
data["X"] = X_train
data["Y"] = Y_train
data["val"] = {}
data["val"]["X"] = X_test
data["val"]["Y"] = Y_test

with open("SkinCancer.pkl", "wb") as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
