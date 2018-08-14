import numpy as np
import cv2
import os
fname = "Recognizer/trainingData.yml"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)
font = cv2.FONT_HERSHEY_SIMPLEX
ids = 0
while(1):
    _,f=img.read()
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(f,(x,y),(x+w,y+h),(0,225,0),3)
        ids,config = recognizer.predict(gray[y:y+h,x:x+w])
        if ids==1 :
            ids="Yogesh"
        elif ids==2 :
            ids="Yashi"
        cv2.putText(f,str(ids),(x,y+h), font, 1, (150,255,0),2)
    cv2.imshow('Live Stream',f)
    if cv2.waitKey(25) == 27:
        break
cv2.destroyAllWindows()
img.release()