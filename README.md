# opencvpr
Hello people.<br/>
This repository contains all the AI tools that i have built and also some that i am currently building.<br/>
I am gonna be explaining each one of these tools and also their underlying algorithms to help you understand them as nicely as possible.<br/><br/>

Before writing about any tool, I would first like to tell you about OpenCV.<br/>
OpenCV is a Open-Source Computer Vision Library which contains tons of functionalities and modules on image and video processing.<br/>
For more details visit https://docs.opencv.org<br/><br/>

#1. Face_Detection
Face detection tools are widely used in modern tech such as self-driving cars, motion sensing, cameras and so on.<br/>
The algorithm used to create the tool is the haar-cascade algorithm.<br/>
Further details are provided in code to help with proper understanding.
About Haar-cascade:<br/>
Haar-Cascade is a speed-based Machine Learning Face-Detection algorithm in which we feed the system with lots of positive and negative data(faces 
and non-faces) and then use some Haar features to detect a face in a grayscale image using some brightness based parameters and applying them in 
different levels to achieve proper detection. Haar-Cascade algorithm technically only works on grayscale images.<br/>
For more details on Haar , go to https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html<br/>

#2. Car_And_Pedestrian_Tracking
This type of tracking algorithms are very widely used in self-driving cars made by Tesla. It is an algorithm with a high scope for optimization, although in the given code it is optimized as much as possible.<br/>
It uses the same underlying algorithm as face detection i.e Haar-Cascade. It uses the same concept as the previous tool , but a different set of trained data.<br/>
Further details are given in code.<br/>
