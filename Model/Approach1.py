#CONST
imageSize, depth = 48, 3
numberOfClasses = 7
filePath = 'data.csv'

import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adamax

TRAIN_END = 28708
TEST_START = TRAIN_END + 1


def split_for_test(list):
    train = list[0:TRAIN_END]
    test = list[TEST_START:]
    return train, test

def processPixels(pixels, img_size=imageSize):

    pixels_list = [item[0] for item in pixels.values.tolist()]

    score_array = []
    for index, item in enumerate(pixels_list):
        data = np.zeros((imageSize, imageSize), dtype=np.uint8)
        pixel_data = item.split()

        for i in range(0, imageSize):
            index = i * imageSize
            data[i] = pixel_data[index:index + imageSize]

        score_array.append(np.array(data))

    score_array = np.array(score_array)
    score_array = score_array.astype('float32') / 255.0
    return score_array


def makeVGG16input(array_input):
    array_input = np.expand_dims(array_input, axis=3)
    array_input = np.repeat(array_input, 3, axis=3)
    return array_input

raw_data = pd.read_csv(filePath)

# convert to one hot vectors
from keras.utils import to_categorical

emotion_array = to_categorical(np.array(raw_data[['emotion']]))

# convert to a 48x48 float matrix
pixel_array = processPixels(raw_data[['pixels']])

# split for test/train
y_train, y_test = split_for_test(emotion_array)
x_train_matrix, x_test_matrix = split_for_test(pixel_array)

x_train_input = makeVGG16input(x_train_matrix)
x_test_input = makeVGG16input(x_test_matrix)

print("Data loading - DONE!")
print("x_train_input ", x_train_input.shape)

# vgg 16. include_top=False so the output is the 512 and use the learned weights
vgg16 = VGG16(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights='imagenet')

for layer in vgg16.layers:
    layer.trainable = False

# build and train model
fialLayer = Sequential()
fialLayer.add(vgg16)
fialLayer.add(Dense(256, input_shape=(512,), activation='relu'))
fialLayer.add(Dense(256, input_shape=(256,), activation='relu'))
fialLayer.add(Dropout(0.5))
fialLayer.add(Dense(128, input_shape=(256,)))
fialLayer.add(Dense(numberOfClasses, activation='softmax'))
fialLayer.summary()
adamax = Adamax()

fialLayer.compile(loss='categorical_crossentropy',
                  optimizer="sgd",
                  metrics=['accuracy'])

# train
history = fialLayer.fit(
    x_train_input, y_train,
    validation_data=(x_test_input, y_test),
    epochs=100,
    batch_size=50)

# Evaluate
score = fialLayer.evaluate(x_test_input,
                           y_test, batch_size=50)

print("After top_layer_model training (test set): {}".format(score))

fialLayer.save('model.h5')
