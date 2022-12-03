import cv2 as cv
import numpy as np
from random import randrange

webcam = cv.VideoCapture(0)

face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_data= cv.CascadeClassifier('Smiles.xml')

while True:
    successful_frame_read, frame = webcam.read()
    
    if(not successful_frame_read):
        print("Webcam Not Working")
        break

    grayscaled_img= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur= cv.GaussianBlur(grayscaled_img, (5,5), 0)

    dilate= cv.dilate(blur, np.ones((3,3)))

    face_coord= face_data.detectMultiScale(dilate)

    for(x,y,w,h) in face_coord:
        cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),5)

        face= frame[y:y+h,x:x+w]

        grayscaled_face= cv.cvtColor(face, cv.COLOR_BGR2GRAY)

        smile_coord= smile_data.detectMultiScale(grayscaled_face,scaleFactor=1.7, minNeighbors=20)

        for (a,b,c,d) in smile_coord:
            cv.rectangle(face,(a,b),(a+c,b+d),(255,0,255),2)
        
        #if len(smile_coord)> 0:
         #   cv.putText(frame, 'smiling',(x,y+h+40),fontScale=3,fontFace=cv.FONT_HERSHEY_PLAIN,color=(255,255,255))
    
    cv.imshow('Smile Please',frame)
    key= cv.waitKey(1)
     
    if(key==81 or key==113):
        break

    """
    for (a,b,c,d) in smile_coord:
            cv.rectangle(face,(a,b),(a+c,b+d),(255,0,255),2)
    

    """


