#Loading Data
from Model.DataProcess import loadData
pixels, emotions = loadData()
print(pixels.shape)
print(emotions.shape)

#Spliting Data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest  = train_test_split(pixels, emotions, test_size = 0.2)

#Loading Models
from Model.CreateModel import createVGG16
vgg16model = createVGG16()

from keras.layers import Input, Flatten, Dense
print("Done")