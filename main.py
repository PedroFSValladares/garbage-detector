import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

model = models.load_model('modelos\\modelo-90porcento.h5')
model.load_weights('modelos\\modelo-90porcento-pesos.h5')
video = cv2.VideoCapture(1)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into dimensions you used while training
        im = im.resize((224, 224))
        img_array = np.array(im)

        #Expand dimensions to match the 4D Tensor shape.
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict function using keras
        prediction = model.predict(img_array)#[0][0]
        predicted_class = np.argmax(prediction)
        print(predicted_class)

        if predicted_class == 0:
                print('vidro')
        if predicted_class == 1:
                print('metal')
        if predicted_class == 0:
                print('papel')
        if predicted_class == 0:
                print('plastico')

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()