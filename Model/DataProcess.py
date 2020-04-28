import pandas as pd
import numpy as np

width, height = 48, 48

def resizeImages(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

def loadData():
    print("Data loading START")
    rawData = pd.read_csv("../data/data.csv")
    pixels = rawData['pixels'].tolist()
    images = []
    for each in pixels:
        image = [int(pixel) for pixel in each.split()]
        image = np.asarray(image).reshape(width, height)
        images.append(image.astype('float32'))
    images = np.asarray(images)
    images = np.expand_dims(images, -1)
    images = np.repeat(images, 3, axis=3)
    emotions = pd.get_dummies(rawData['emotion'])
    print(emotions)
    images = resizeImages(images)
    print("Data loading DONE")
    return images, emotions


def generate_dataset():
    """generate dataset from csv"""

    df = pd.read_csv("../data/data.csv")

    train_samples = df[df['Usage'] == "Training"]
    validation_samples = df[df["Usage"] == "PublicTest"]
    test_samples = df[df["Usage"] == "PrivateTest"]

    #y_train = train_samples.emotion.astype(np.int32).values
    y_train = pd.get_dummies(train_samples['emotion'])
    #y_valid = validation_samples.emotion.astype(np.int32).values
    y_valid = pd.get_dummies(validation_samples['emotion'])
    #y_test = test_samples.emotion.astype(np.int32).values
    y_test = pd.get_dummies(test_samples['emotion'])

    X_train = np.array([np.fromstring(image, np.uint8, sep=" ").reshape((48, 48)) for image in train_samples.pixels])
    X_valid = np.array(
        [np.fromstring(image, np.uint8, sep=" ").reshape((48, 48)) for image in validation_samples.pixels])
    X_test = np.array([np.fromstring(image, np.uint8, sep=" ").reshape((48, 48)) for image in test_samples.pixels])

    X_train = X_train.reshape((-1, 48, 48, 1)).astype(np.float32)
    X_valid = X_valid.reshape((-1, 48, 48, 1)).astype(np.float32)
    X_test = X_test.reshape((-1, 48, 48, 1)).astype(np.float32)

    X_train = np.repeat(X_train, 3, axis=3)
    X_valid = np.repeat(X_valid, 3, axis=3)
    X_test = np.repeat(X_test, 3, axis=3)

    X_train_std = X_train / 255.
    X_valid_std = X_valid / 255.
    X_test_std = X_test / 255.

    return X_train_std, y_train, X_valid_std, y_valid, X_test_std, y_test