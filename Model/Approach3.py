import numpy as np
import pandas as pd
from keras.utils import to_categorical

# CONST
imageSize, depth = 48, 1
numberOfClasses = 7
filePath = '/content/drive/My Drive/Colab Notebooks/data.csv'

TRAIN_END = 28708
TEST_START = TRAIN_END + 1


def split_for_test(list):
    train = list[0:TRAIN_END]
    test = list[TEST_START:]
    return train, test


def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list


def processPixels(pixels, img_size=imageSize):
    pixels_list = pandas_vector_to_list(pixels)

    score_array = []
    for index, item in enumerate(pixels_list):
        data = np.zeros((imageSize, imageSize), dtype=np.uint8)
        pixel_data = item.split()

        for i in range(0, imageSize):
            index = i * imageSize
            data[i] = pixel_data[index:index + imageSize]

        score_array.append(np.array(data))

    score_array = np.array(score_array)
    score_array = np.expand_dims(score_array, axis=3)
    score_array = score_array.astype('float32') / 255.0

    return score_array


# K.set_learning_phase(0)

raw_data = pd.read_csv(filePath)

# convert to one hot vectors

emotion_array = to_categorical(np.array(raw_data[['emotion']]))

# convert to a 48x48 float matrix
pixel_array = processPixels(raw_data[['pixels']])

# split for test/train
y_train, y_test = split_for_test(emotion_array)
x_train_matrix, x_test_matrix = split_for_test(pixel_array)

print("Data loading - DONE!")
print("x_train_matrix ", x_train_matrix.shape)

import matplotlib.pyplot as plt
imgNo = 20000
img = np.repeat(x_train_matrix[imgNo], 3, axis = 2)
plt.imshow(img)
plt.show()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# build and train model
model = Sequential()
model.add(Conv2D(64, 5, activation='relu', input_shape=(imageSize, imageSize, depth)))
model.add(MaxPooling2D(3, strides=2))
model.add(Conv2D(64, 5, activation="relu"))
model.add(MaxPooling2D(3, strides=2))
model.add(Conv2D(128, 4, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(3072, activation="relu"))
model.add(Flatten())
model.add(Dense(numberOfClasses, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# train
history = model.fit(
    x_train_matrix, y_train,
    validation_data=(x_test_matrix, y_test),
    epochs=100,
    batch_size=50)

# Evaluate
score = model.evaluate(x_test_matrix,
                       y_test, batch_size=50)

print("After top_layer_model training (test set): {}".format(score))

model.save('model.h5')

