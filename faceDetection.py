from __future__ import print_function
import cv2

faceCascade = cv2.CascadeClassifier('C:/Gabor/Programs/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

if faceCascade.empty():
    print('--(!)Error loading face cascade')
    exit(0)

# Capture framse from a camera
cap = cv2.VideoCapture(0)

while True:

    # Reads frames from a camera 
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