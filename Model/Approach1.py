#Constants
width, height, depth = 48, 48, 3
numOfClasses = 7
epochs = 2
batch_size = 50

#Loading Data
from Model.DataProcess import loadData
pixels, emotions = loadData()
print(pixels.shape)
print(emotions.shape)

#Spliting Data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(pixels, emotions, test_size = 0.2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

#Loading VGG16 Model
from keras.applications.vgg16 import VGG16
vgg16model = VGG16(include_top=False, weights='imagenet',
                 input_shape=(width, height, depth), pooling='avg')

#Creating final classifier
from keras.models import Sequential
from keras.layers import Dropout, Dense

myModel = Sequential()
myModel.add(Dense(128, input_shape=vgg16model.output_shape[1:], activation="relu"))
myModel.add(Dense(256, input_shape=(256,), activation="relu"))
myModel.add(Dropout(0.25))
myModel.add(Dense(128, input_shape=(256,)))
myModel.add(Dense(numOfClasses, activation="softmax"))

#Creating final model with both of them
from keras.models import Model
finalModel = Model(input=vgg16model.input,
                   outputs=myModel(vgg16model.output))

#Creating optimizer
from keras.optimizers import Adamax
adamax = Adamax()

finalModel.compile(loss='categorical_crossentropy',
                   optimizer=adamax,
                   metrics=['accuracy'])

finalModel.summary()

#Extenging traingn data
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,
    zoom_range = 0.05)  # zoom images in range [1 - zoom_range, 1+ zoom_range]
datagen.fit(xtrain)

#Saving best model
import os
model_file = os.path.join(os.getcwd(), 'cnn_model.{epoch:03d}.h5')

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath=model_file,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)
callbacks = [checkpoint]

#Fiting
history = finalModel.fit_generator(
    datagen.flow(xtrain, ytrain, batch_size=batch_size),
    steps_per_epoch=xtrain.shape[0]/32,
    epochs=2,
    validation_data=(xtest, ytest),
    callbacks = callbacks
)

#Saving hisotry
import pandas as pd
pd.DataFrame(history.history).to_csv("history.csv")

print("Done")