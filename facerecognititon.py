import cv2
import numpy as np

face1=np.load('gopal.npy').reshape(100,50*50*3)

face2=np.load('adarsh.npy').reshape(100,50*50*3)

data=np.concatenate([face1,face2])

dataset = cv2.CascadeClassifier('hr.xml')

capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX

user={0:"gopal",1:"adarsh"}

labels = np.zeros((200,1))

labels[:100,:]=0.0

labels[100:,:]=1.0

def arr_sqrt (x1,x2):
       return np.sqrt(sum((x2-x1)**2))

def knn(x,train,k=5):
    n=train.shape[0]
    arr=[]
    for i in range(n):
        arr.append(arr_sqrt(x,train[i]))
    arr=np.array(arr)
    sortindex=np.argsort(arr)
    lab=labels[sortindex][:k]
    count=np.unique(lab,return_counts=True)
    return count[0][np.argmax(count[1])]

while True:
    ret,img = capture.read()
    if ret:
        #img=cv2.resize(img,None,fx=0.1,fy=0.1)     --for resize window
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)

        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
            myface=img[y:y+h,x:x+w,:]
            myface=cv2.resize(myface,(50,50))
            label=knn(myface.flatten(),data)
            user_name=user[int(label)]
            
            cv2.putText(img,user_name,(x,y),font,1,(0,255,0),2)
        cv2.imshow('result',img)
        if cv2.waitKey(1)==27:
            break
    else:print("camera not working")
capture.release()
cv2.destroyAllWindows()
