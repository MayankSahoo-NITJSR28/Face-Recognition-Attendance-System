# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:03:39 2026

@author: hp
"""

# ip webcam -> Used to collect data 

import cv2
import numpy as np
import os

#automatically create image folder

os.makedirs("images",exist_ok=True)

# use har_cascade to extract face

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#open_laptop_camera at 0 index for def

cap = cv2.VideoCapture(0)

data = [] # Store the 100 captured face image

# loop until 100 photo captured

while len(data) < 100:
    # read frame from camera
    ret,frame = cap.read()
    if not ret:
        print("Fail to access your camera")
        break
    # Detect face model load
    face_points = classifier.detectMultiScale(frame,1.3,5)
    
    # if at leat one image detected
    if len(face_points)>0:
        for x,y,w,h in face_points:
            #crop the detect face
            face_frame = frame[y:y+h+1,x:x+w+1] # 1 -> adds one
            #show only the face
            cv2.imshow("Only_face",face_frame)
            
            # add face to data list
            
            if len(data) < 100:
                print(len(data)+1,"/100")
                data.append(face_frame)
                break
    
    # display count img on mainframe
    cv2.putText(frame, str(len(data)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)
    cv2.imshow("Frame",frame)
    
    # adding a waitkey
    
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release() # auto camera off

cv2.destroyAllWindows()  # close all open cv windows

# if exact 100 samples collected

if len(data) == 100:
    name = input("Enter face holder name: ")
    
    #save format
    for i in range(100):
        cv2.imwrite("images/"+ name + "_" + str(i) + ".jpg", data[i])
    print("Image Stored")
else:
    print("Need more Data")
        