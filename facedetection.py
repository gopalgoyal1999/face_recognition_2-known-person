import cv2
import numpy as np
dataset = cv2.CascadeClassifier('hr.xml')
capture = cv2.VideoCapture(0)
data=[]

while True:
    ret,img = capture.read()
    if ret:
        #img=cv2.resize(img,None,fx=0.1,fy=0.1)     --for resize window
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)

        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
            myfaces=img[y:y+h,x:x+w,:]
            myfaces=cv2.resize(myfaces,(50,50))
            if len(data) < 100:
                data.append(myfaces)
                print(len(data))
        cv2.imshow('result',img)
        if cv2.waitKey(1)==27 or len(data)>=100:
            break
    else:
        print("camera not working")

face=np.array(data)
np.save("gopal.npy",face)
capture.release()
cv2.destroyAllWindows()
