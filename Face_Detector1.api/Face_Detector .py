import cv2 as cv 
from random import randrange

# Add some trained face data 
trained_face_data= cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture webcam
webcam =cv.VideoCapture(0)

# Run all the frames in the webcam
while True:
    successful_frame_read, frame= webcam.read()

    grayscaled_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    # Check face coordinates, detectMultiScale means it can detect faces of all sizes with increased sensitivity.
    face_coord = trained_face_data.detectMultiScale(grayscaled_frame,1.0485258,6)

    # Draw rectangles around face in every frame.
    for (x,y,w,h) in face_coord:
        cv.rectangle(frame,(x,y), (x+w,y+h),(randrange(256),randrange(256),randrange(256)), 5) 

    #display the current frame
    cv.imshow('Crazy coder', frame)

    key = cv.waitKey(1)
    if (key==81 or key==113):
        break
webcam.release()


    
    
"""
# Make it wait after every frame 
cv.waitKey(1)


# Add image 
img=cv.imread('png-transparent-robert-downey-jr-robert-downey-jr-iron-man-hollywood-actor-robert-downey-jr-pic-celebrities-hair-formal-wear-thumbnail.png')

# Must convert to Grayscale
grayscaled_img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Check face coordinates, detectMultiScale means it can detect faces of all sizes.
face_coord = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coord)

for (x,y,w,h) in face_coord:
    cv.rectangle(img,(x,y), (x+w,y+h),(randrange(256),randrange(256),randrange(256)), 5) 

#Display Image
cv.imshow('Crazy Coder', img)
cv.waitKey()
print("Code Completed")

About Harcascade:
Haar-Cascade is a speed-based Machine Learning Face-Detection algorithm in which we feed the system with lots of positive and negative data(faces 
and non-faces) and then use some Haar features to detect a face in a grayscale image using some brightness based parameters and applying them in 
different levels to achieve proper detection. Haar-Cascade algorithm technically only works on grayscale images.

For more details on Haar , go to https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

Face detection is a widely-used and very useful tool used by self-driving cars and cameras all over the world  
"""