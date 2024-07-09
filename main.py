
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture("rtsp://192.168.226.201:554/profile1")

# cap = cv2.VideoCapture(0)
myClassifier = cvzone.ClassificationModule.Classifier('Mymodels/keras_model.h5', 'Mymodels/labels.txt')

while True:
    _, img = cap.read()
    predictions = myClassifier.getPrediction(img)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)

