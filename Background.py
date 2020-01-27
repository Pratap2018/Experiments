from threading import Thread
import os
import cv2
import numpy as np 
print(os.getcwd())
try:
    os.mkdir('dataset_bgr_rmv')
except OSError:
    print("exception")
    
def capture(x):
    try:
        os.mkdir('dataset_bgr_rmv/'+x)
        print(os.getcwd())
    except:
        print("error")
    
    cap=cv2.VideoCapture(0)
    fgbg=cv2.createBackgroundSubtractorMOG2()
    knn=cv2.createBackgroundSubtractorKNN()
    i=len(os.listdir('dataset_bgr_rmv/'+x))
    while True:
        ret,frame=cap.read()
        frame1=cv2.flip(frame,1)

        frame=frame1[1:350,289:638]
        fg=fgbg.apply(frame)
        kn=knn.apply(frame)
        cv2.rectangle(frame1,(288,0),(639,351),(0,255,0),1)
        
        cv2.imshow("cap",frame1)
        cv2.imshow('fg',fg)
        
        cv2.imshow("knn",kn)
        
        k=cv2.waitKey(5) & 0xff 
        if k==ord(' '):
            cv2.imwrite('dataset_bgr_rmv/'+x+'/'+x+'_'+str(i)+'.jpg',kn)    
            i+=1
        if k==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
   
def inp():
    x=input()
    return x
if __name__ == "__main__":
    
    x=inp()
    capture(x)