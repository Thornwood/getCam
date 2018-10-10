#import os
#import sys
import cv2

#cascPath = sys.argv[0]
#cascPath = os.getcwd()+'/haarcascades/haarcascade_frontalface_default.xml'
#cascPath1 = os.getcwd()+'/haarcascades/haarcascade_frontalface_alt.xml'
#cascPath2 = os.getcwd()+'/haarcascades/haarcascade_frontalface_alt2.xml'
#cascPath3 = os.getcwd()+'/haarcascades/haarcascade_frontalface_alt_tree.xml'

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceCascade.empty()

# Capture framse from a camera
cap = cv2.VideoCapture(0)

while True:

    # reads frames from a camera 
    ret, frame = cap.read()

    # Display grayscale image
    if ret != 0:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayscale, 1.3, 5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

     # Wait for Esc key to stop 
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break

# Close the window
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()