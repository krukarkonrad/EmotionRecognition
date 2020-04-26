from keras.applications.vgg16 import VGG16

def createVGG16():
    return VGG16(include_top=False, weights='imagenet',
                 input_shape=(48, 48, 3), pooling='avg')

##createVGG16().summary()

