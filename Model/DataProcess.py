import pandas as pd
import numpy as np
width, height = 48, 48

def resizeImages(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

def loadData():
    rawData = pd.read_csv("./data/data.csv")
    pixels = rawData['pixels'].tolist()
    images = []
    for each in pixels:
        image = [int(pixel) for pixel in each.split()]
        image = np.asarray(image).reshape(width, height)
        images.append(image.astype('float32'))
    images = np.asarray(images)
    images = np.expand_dims(images, -1)
    emotions = pd.get_dummies(rawData['emotion'])
    images = resizeImages(images)
    return images, emotions