# Import Libraries
import cv2
import numpy as np

# Input the name of the person
file_name = input('Enter the name of the file: ')
file_format = input('Enter the format of the file: ')

# Read Image 
img = cv2.imread('images/'+file_name+'.'+file_format)

# Face Detection
eye_cascade = cv2.CascadeClassifier('third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('third-party/frontalNose18x15.xml')

# Detect and store eyes in the array
eyes = eye_cascade.detectMultiScale(img,1.3,5)
# Detect and store nose in the array
nose = nose_cascade.detectMultiScale(img,1.3,5)

# Draw a bounding box around the largest face (last face is largest)
for (x,y,w,h) in eyes:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# Draw a bounding box around the largest face (last face is largest)
for (x,y,w,h) in nose:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# Display the video output
cv2.imshow('Frame', img)

# Release all system resources used
cv2.waitKey(0)
cv2.destroyAllWindows()