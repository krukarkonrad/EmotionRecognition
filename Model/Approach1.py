#Const
width, height, depth = 48, 48, 3
numOfClasses = 7

#Load data
import pandas as pd
dataFile = pd.read_csv('data.csv')
print(dataFile.shape)
trainSamples = dataFile[dataFile['Usage'] == "Training"]
testSamples = dataFile[dataFile["Usage"] == "PrivateTest"]
validationSamples = dataFile[dataFile["Usage"] == "PublicTest"]

#Load image
import numpy as np
pixelMatrix = np.array([int(string)/255 for string in trainSamples.iloc[0,1].split(' ')])
pixelMatrix = pixelMatrix.reshape(width, height)

#Show image
import matplotlib.pyplot as plt
plt.imshow(pixelMatrix, cmap='gray');

#Prepare data for model
def prepareData(dataSet):
  #1D -> 2D
  xTrain = []
  for i in range(len(dataSet)):
    pixelMatrix = np.array([int(string) for string in dataSet.iloc[i,1].split(' ')])
    pixelMatrix = pixelMatrix.reshape(width, height)
    xTrain.append(pixelMatrix)

  x = np.array(xTrain)
  yTrain = dataSet.iloc[:,0]

  #Categorize categories to be compatible with softmax activation
  from keras.utils import to_categorical
  y = to_categorical(np.array(yTrain))
  return x, y

xTrain, yTrain = prepareData(trainSamples)
print(xTrain.shape)
print(yTrain.shape)

xTest, yTest = prepareData(testSamples)
print(xTest.shape)
print(yTest.shape)

xVal, yVal = prepareData(validationSamples)
print(xVal.shape)
print(yVal.shape)

#Reshape to 4D, as VGG16 requires
xTrainReshaped = np.expand_dims(xTrain, axis = 3)
xTestReshaped = np.expand_dims(xTest, axis = 3)
xValReshaped = np.expand_dims(xVal, axis = 3)

#Repeat to get (:, :, :, 3)
xTrainReshaped = np.repeat(xTrainReshaped, 3, axis = 3)
xTestReshaped = np.repeat(xTestReshaped, 3, axis = 3)
xValReshaped = np.repeat(xValReshaped, 3, axis = 3)

print(xTrainReshaped.shape)
print(yTrain.shape)
print(xTestReshaped.shape)
print(yTest.shape)
print(xValReshaped.shape)
print(yVal.shape)

#Import VGG16 without top layers
from keras.applications.vgg16 import VGG16
baseModel = VGG16(include_top= False, weights='imagenet', input_shape = (width,height,depth))

baseModel.summary()

print(baseModel.predict([[xTrainReshaped[0]]]).shape)

#Create own model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

myModel = Sequential()

myModel.add(baseModel)
myModel.add(Flatten())
myModel.add(Dense(512, input_shape=(512,), activation='relu'))
myModel.add(Dense(256, input_shape=(256,),activation='relu'))
myModel.add(Dropout(0.5))
myModel.add(Dense(128, input_shape=(256,)))
myModel.add(Dense(7, activation='softmax'))

myModel.summary()

myModel.compile(loss='categorical_crossentropy',
                optimizer="sgd",
                metrics=['accuracy'])

for layer in myModel.layers[:1]:
   layer.trainable=False
for layer in myModel.layers[1:]:
   layer.trainable=True

for layer in myModel.layers:
  print(layer.name)
  print(layer.trainable)

#Fit params
noEpochs = 50
batchSize = 100
history = myModel.fit(xTrainReshaped , yTrain,
                      epochs=noEpochs, batch_size=batchSize,
                      validation_data=(xValReshaped, yVal))

myModel.save("test.h5")

totest = 2
print(myModel.predict([[xTrainReshaped[totest]]])*10)
print(yTest[totest])