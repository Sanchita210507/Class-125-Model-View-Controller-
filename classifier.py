import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X, y = fetch_openml('mnist_784',version = 1, return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=9,train_size = 7500, test_size = 2500)
# scaling the features
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scaled, y_train)

def get_prediction(image):
    impil = Image.open(image)
    imgbw = impil.convert('L')
    imgbwresized = imgbw.resize((28,28),Image.ANTIALIAS)
    pixelFilter = 20
    # Converting to scalar quantity
    minPixel = np.percentile(imgbwresized, pixelFilter)
    # limting the values between 0 and 255
    imgbwscaled = np.clip(imgbwresized-minPixel,0,255)
    maxPixel = np.max(imgbwresized)
    imgbwscaled = np.asarray(imgbwscaled)/maxPixel
    testSample = np.array(imgbwscaled).reshape(1,784)
    testPred = clf.predict(testSample)
    return testPred[0]