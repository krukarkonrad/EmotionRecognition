from keras.models import model_from_json
# from PIL import Image
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
import cv2
import sys
# Emotions dictionary
emotions = {"anger" : 0,
"disgust" : 1,
"fear" : 2,
"happy" : 3,
"sad" : 4,
"surprise" : 5,
"neutral" : 6}
from keras.models import load_model
loaded_model = load_model('model.h5')
cap = cv2.VideoCapture(0)
# Get user supplied values
# imagePath = sys.argv[1]
# cascPath = sys.argv[2]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Conver to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face Detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x ,y ,w ,h) in faces:
        crop_img = gray[y: y +h, x: x +w]

        # Get width and height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize for our model (48x48x1)
        small = cv2.resize(crop_img, dsize = (48 ,48))
        # convert size from 48x48 to 1x48x48
        image3D = np.expand_dims(small ,axis = 0)
        # convert to 1x48x48x1
        image4D = np.expand_dims(image3D, axis = 3)
        # convert to 1x48x48x3
        image4D3 = np.repeat(image4D, 3, axis=3)
        print(image3D.shape)
        # Model each frame
        emotions_prob = loaded_model.predict(image4D3)[0]
        print(emotions_prob)
        # Convert emotion probabilities into binary, where 1 is the emotion you're feeling
        listt = [1 if metric == emotions_prob.max() else 0 for metric in emotions_prob]
        # Get the index 1 in the binary list, listt
        emotion_index = listt.index(1)
        emotion = list(emotions.keys())[emotion_index]

        # Show Emotion on Video
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_placement  = (int(width /2 - 500) ,int(height /2 + 100))
        fontScale = 1
        fontColor = (255 ,255 ,255)
        lineType = 4

        cv2.putText(frame,
                    '{}'.format(emotion),
                    text_placement,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # Display the resulting frame
    cv2.imshow('frame' ,frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

