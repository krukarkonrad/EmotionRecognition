#Constants
width, height, depth = 48, 48, 3
numOfClasses = 7
epochs = 50
batch_size = 30

#Loading Data
from Model.DataProcess import loadData
pixels, emotions = loadData()
#print(pixels.shape)
#print(emotions.shape)

#Spliting Data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(pixels, emotions, test_size = 0.2)
#print(xtrain.shape)
#print(xtest.shape)
#print(ytrain.shape)
#print(ytest.shape)

#Loading VGG16 Model
from keras.applications.vgg16 import VGG16
vgg16model = VGG16(include_top=False, weights='imagenet',
                 input_shape=(width, height, depth), pooling='avg')

#vgg16model.summary()

#Frezzing layers at VGG16 model - not trainable
for layer in vgg16model.layers:
    layer.trainable = False

#Creating final classifier
from keras.models import Sequential
from keras.layers import Dropout, Dense

myModel = Sequential()
myModel.add(vgg16model)
myModel.add(Dense(128, input_shape=(512,), activation="relu"))
myModel.add(Dense(256, input_shape=(256,), activation="relu"))
myModel.add(Dropout(0.25))
myModel.add(Dense(128, input_shape=(256,)))
myModel.add(Dense(numOfClasses, activation="softmax"))
#myModel.summary()

#Creating optimizer
from keras.optimizers import Adamax
adamax = Adamax()

myModel.compile(loss='categorical_crossentropy',
                   optimizer=adamax,
                   metrics=['accuracy'])

myModel.summary()

#Creating callback
import os
model_file = os.path.join(os.getcwd(), 'cnn_model.{epoch:03d}.h5')

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath=model_file,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)
callbacks = [checkpoint]

#Fiting model
history = myModel.fit(
    xtrain, ytrain,
    epochs=epochs,
    validation_data=(xtest, ytest),
    callbacks = callbacks
)

#Saving hisotry
import pandas as pd
pd.DataFrame(history.history).to_csv("history.csv")

myModel.save("obyKoniec.h5")

print("Done")