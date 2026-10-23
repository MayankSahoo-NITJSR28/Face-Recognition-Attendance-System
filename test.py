# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:49:09 2026

@author: hp
"""

import cv2
import numpy as np
from  keras.models import load_model

classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

model = load_model("Final_Model.h5")

# Camera access

cap = cv2.VideoCapture(0)

# Add labels
def get_pred_label(pred):
    labels=["Mayank"]
    return labels[pred]

#defpreprocess
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100,100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1,100,100,1)
    img = img/255.0
    return img

# main loop for detection
ret = True
while ret:
    ret,frame = cap.read()
    faces = classifier.detectMultiScale(frame,1.5,5)
    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        #draw rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),5)
        prediction = np.argmax(model.predict(preprocess(face),verbose=0))
        # display pred name
        cv2.putText(frame, get_pred_label(prediction),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(123,56,34),2)
        
    # show webcam feed
    cv2.imshow("Capture",frame)
    # press q
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

        

