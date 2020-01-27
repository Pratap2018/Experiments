import os
import cv2
import model_trained
import model_bw
import torch

cap=cv2.VideoCapture(0)
knn=cv2.createBackgroundSubtractorKNN()
device = torch.device('cpu')
modelx = model_bw.FNet()
modelx.load_state_dict(torch.load('mode.pth', map_location=device))
print(modelx)
while True:
    ret,frame=cap.read()
    frame1=cv2.flip(frame,1)

    frame=frame1[1:350,289:638]
    #fg=fgbg.apply(frame)
    kn=knn.apply(frame)
    cv2.rectangle(frame1,(288,0),(639,351),(0,255,0),1)
      
    
    #cv2.imshow('fg',fg)
    s=s=torch.from_numpy(kn)
    s=s.type(torch.float32)
    s=s.unsqueeze(0)
    s=s.unsqueeze(0)
    l=torch.argmax(modelx(s),1).item()+1
    frame=cv2.putText(frame1,str(l),(200,200),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),cv2.LINE_4)
    cv2.imshow("cap",frame)
    cv2.imshow("knn",kn)
    k=cv2.waitKey(1)
    if k==ord('q') & 0xff:
            break
cap.release()
cv2.destroyAllWindows()