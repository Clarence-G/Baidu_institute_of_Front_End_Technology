import cv2

cap = cv2.VideoCapture(0)
while(1):
    i = 1
    ret, frame = cap.read()
    cv2.imshow("capture", frame)
    # get 100 pcitures one number,named by '1.jpg' '2.jpg' and so on
    while i<=100:        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("xxxx" + str(i) + '.jpg', frame)   # write your path here
    break
cap.release()
cv2.destroyAllWindows()
