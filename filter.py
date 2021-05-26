# Import Libraries
import cv2
import numpy as np

# Input the name of the person
file_name = input('Enter the name of the file: ')
file_format = input('Enter the format of the file: ')

# Read Image 
img = cv2.imread('images/'+file_name+'.'+file_format)

# Read Filters
glasses = cv2.imread('filters/glasses.png', -1)
mustache = cv2.imread('filters/mustache.png', -1)

# Face Detection
eye_cascade = cv2.CascadeClassifier('third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('third-party/frontalNose18x15.xml')

# Detect and store eyes in the array
eyes = eye_cascade.detectMultiScale(img,1.3,5)
# Detect and store nose in the array
nose = nose_cascade.detectMultiScale(img,1.3,5)

# set dimension
x,y,w,h = eyes[0]
width = int(1.1 * glasses.shape[1] * w / glasses.shape[1])
height = int(1.1 * glasses.shape[0] * w / glasses.shape[1])
dim = (width, height)
# resize filter
glasses = cv2.resize(glasses, dim, interpolation = cv2.INTER_AREA)
# Add offset
y1, y2 = y-int(y*0.05), y-int(y*0.05)+glasses.shape[0]
x1, x2 = x-int(x*0.05), x-int(x*0.05)+glasses.shape[1]
# Remove alpha channel from filter
alpha_s = glasses[:, :, 3] / 255.0
alpha_l = 1.0 - alpha_s
# Add filter
for c in range(0, 3):
    img[y1:y2, x1:x2, c] = (alpha_s * glasses[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])

# Set dimensions
x,y,w,h = nose[0]
width = int(mustache.shape[1] * w / mustache.shape[1])
height = int(mustache.shape[0] * w / mustache.shape[1])
dim = (width, height)
# resize filter
mustache = cv2.resize(mustache, dim, interpolation = cv2.INTER_AREA)
# Add offset
y1, y2 = y+round(h*(1/2)), y+round(h*(1/2))+mustache.shape[0]
x1, x2 = x, x+mustache.shape[1]
# Remove alpha channel from filter
alpha_s = mustache[:, :, 3] / 255.0
alpha_l = 1.0 - alpha_s
# Add filter
for c in range(0, 3):
    img[y1:y2, x1:x2, c] = (alpha_s * mustache[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])

# Display the video output
cv2.imshow('Frame', img)
cv2.imwrite('images/test.jpg',img)

# Release all system resources used
cv2.waitKey(0)
cv2.destroyAllWindows()