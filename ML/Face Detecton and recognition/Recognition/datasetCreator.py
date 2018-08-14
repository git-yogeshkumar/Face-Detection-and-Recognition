import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.VideoCapture(0)
id = input('Enter User Id')
sampleNo = 0
while(1):
    _,f=img.read()
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        sampleNo = sampleNo+1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNo)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(f,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = f[y:y+h, x:x+w]
        cv2.waitKey(100)
    cv2.imshow('Live Stream',f)
    cv2.waitKey(1)
    if (sampleNo>50):
        break
img.release()
cv2.destroyAllWindows()
