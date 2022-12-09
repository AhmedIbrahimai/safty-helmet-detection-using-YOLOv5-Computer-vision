import cv2
import torch
import numpy as np
import os
os.chdir(r"C:\Users\amb\Downloads\safty-helmet\yolov5safetyhelmet")
#import sys
#sys.setrecursionlimit(3000)

model = torch.hub.load('ultralytics/yolov5', 'custom','best.pt', force_reload=True)


cap=cv2.VideoCapture('helmet.mp4')


    

#count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    #count += 1
    #if count % 3 != 0:
        #continue
    frame=cv2.resize(frame,(1020,600))
    results=model(frame)
    frame = np.squeeze(results.render())
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1):
        break
cap.release()
cv2.destroyAllWindows()



#@lru_cache(128)
