import cv2



cap = cv2.VideoCapture("rtsp://192.168.226.201:554/profile1")

# cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()