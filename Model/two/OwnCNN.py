#CONST
imageSize, depth = 48, 3
numberOfClasses = 7
filePath = '/content/drive/My Drive/Colab Notebooks/data.csv'

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping

TRAIN_END = 28708
TEST_START = TRAIN_END + 1

def split_for_test(list):
    train = list[0:TRAIN_END]
    test = list[TEST_START:]
    return train, test

def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list

def processPixels(pixels):
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
    score_array = score_array.astype('float32') / 255.0
    score_array = np.expand_dims(score_array, axis=3)
    return score_array

raw_data = pd.read_csv(filePath)

# Convert to one hot vectors
from keras.utils import to_categorical

emotion_array = to_categorical(np.array(raw_data[['emotion']]))

# Convert to a 48x48 float matrix
pixel_array = processPixels(raw_data[['pixels']])

# Split for test/train
y_train, y_test = split_for_test(emotion_array)
x_train_matrix, x_test_matrix = split_for_test(pixel_array)

# Build and train model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                 input_shape = (48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numberOfClasses, activation='softmax'))
model.summary()

adam = Adamax()

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20)

# train
history = model.fit(x_train_matrix, y_train, batch_size=128, epochs=500,
          validation_data=(x_test_matrix, y_test), shuffle=True,
          callbacks=[early_stopping])

# Evaluate
score = model.evaluate(x_test_matrix,
                           y_test, batch_size=50)

print("After model training (test set): {}".format(score))

model.save('model.h5')
pd.DataFrame(history.history).to_csv("history.csv")