import cv2 as cv
import numpy as np
from random import randrange

webcam = cv.VideoCapture('istockphoto-145754478-640_adpp_is.mp4  ')

# Classifier file
classifier_file= 'cars.xml'
car_data = cv.CascadeClassifier(classifier_file)
body_data= cv.CascadeClassifier('opencv_haarcascade_fullbody.xml')

while True:
    successful_frame_read , frame = webcam.read()

     # Convert into grayscale
    grayscale_img= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    grayscale_body= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    # Gaussian Blur
    blur= cv.GaussianBlur(grayscale_img,(5,5),0)
    body_blur= cv.GaussianBlur(grayscale_body,(5,5),0)

    # Dilation
    dilate= cv.dilate(blur,((3,3)))
    body_dilate= cv.dilate(body_blur,((3,3)))

    car_coord = car_data.detectMultiScale(dilate)
    body_coord= body_data.detectMultiScale(body_dilate)

    # Draw rectangles around pedestrians and cars
    for (x,y,w,h) in car_coord:
       cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)

    for (x,y,w,h) in body_coord:
       cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),20)
    
    cv.imshow('Car_And_Pedestrian_Tracking', frame)
    key= cv.waitKey(1)
    if(key== 81 or key==113):
        break
webcam.release()






"""
 # For image and classifier files.
img_file= 'rbdowney.png'
classifier_file= 'cars.xml'

 #Read the image and read the trained data.
img = cv.imread(img_file)
car_data = cv.CascadeClassifier(classifier_file)

 # Convert into grayscale
grayscale_img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Gaussian Blur
blur= cv.GaussianBlur(grayscale_img,(5,5),0)

# Dilation
dilate= cv.dilate(blur,((3,3)))

car_coord = car_data.detectMultiScale(dilate)

cnt=0
for (x,y,w,h) in car_coord:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
    cnt= cnt+1

print(cnt)

cv.imshow('Yayyyy', img )
key= cv.waitKey(0)

About Harcascade:
Haar-Cascade is a speed-based Machine Learning Face-Detection algorithm in which we feed the system with lots of positive and negative data(faces 
and non-faces) and then use some Haar features to detect a face in a grayscale image using some brightness based parameters and applying them in 
different levels to achieve proper detection. Haar-Cascade algorithm technically only works on grayscale images.

Gaussian blur:
Gaussian blur is one of the techniques of image processing. 
It is widely used in graphics designing too for reducing the noise and smoothing the image so that for further preprocessing, 
it will generate better output. 
Along with reducing the noise in the image Gaussian blur technique also reduces the image’s details.

Dilation:
Dilation is one of the morphological techniques where we try to fill the pixels with the element, also known as kernels (structured pieces),
to fill the missing parts of the images whenever needed.

For more details on Haar , go to https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

Face detection is a widely-used and very useful tool used by self-driving cars and cameras all over the world  


"""