import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

number_of_classes=5
dataset_size=100

labels=['Chutt']

cap=cv2.VideoCapture(0)
for label in labels:
    if not os.path.exists(os.path.join(DATA_DIR,label)):
        os.makedirs(os.path.join(DATA_DIR,label))

    print('Collecting data for class {}'.format(label))

    done=False
    while True:
        ret,frame=cap.read()
        cv2.putText(frame,'Ready? Press "Q" !',(100,50),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,0),3,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(25)==ord('q'):
            break

    counter=0
    while counter < dataset_size:
        ret,frame=cap.read()
        cv2.imshow('frame',frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR,label,'{}.jpg'.format(counter)),frame)
        counter+=1

cap.release()
cv2.destroyAllWindows()
