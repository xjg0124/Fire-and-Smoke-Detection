# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:46:47 2016

@author: xjg
"""
import numpy as np
import cv2
from collections import deque
from Smoke_Tool import Smoke_tool
from smoke_classify import smoke_classify
import caffe

def Smoke_detector(VideoPath):
    
    # model
    model_def_file = './model/0.991/conv3d_smoke_deploy.prototxt'
    model_file = './model/0.991/conv3d_smoke_iter_52000'
    mean_file = './model/0.99/smoke_train_mean.binaryproto'
    net = caffe.Net(model_def_file, model_file)

    # caffe init
    gpu_id = 0
    net.set_device(gpu_id)
    net.set_mode_gpu()
    net.set_phase_test()
    
    cap = cv2.VideoCapture(VideoPath)
    while(cap.isOpened()):
         WT = 16
         WD = 24
         Smoketool = Smoke_tool()
         frame_count = 0
         frames = deque()

         success, img = cap.read()
    
         gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
         img_back_gray = gray_img.copy()

         rows = img.shape[1]
         cols = img.shape[0]

         locate_list = Smoketool.ImageBlock(rows,cols,WD)     
         
         VideoIno = open('./VideoInformation','a')
         first_smoke_frame = 0
         fsflag = True
         error_frame = 0
         M_blocks = []
         delay = 2
         
         Smoketool.OpticalFlowPara(img)
         while success:
             success, img = cap.read()
             if success != 1:
                 break
             frame_count+=1
             gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
             
             #Motion_blocks = Smoketool.MotionDetection(gray_img,img_back_gray,locate_list,WD)
             #Motion_blocks = Smoketool.ColorDetection(img,locate_list,WD)
             Motion_blocks = Smoketool.MotionDetectionOpticalFlow(gray_img,img_back_gray,WD)
             img_back_gray = gray_img.copy()
        
                          
             if len(frames) < WT:
                 frames.append(img.copy())
                 
             if len(frames) == WT:
                 
                 smoke_blocks = list()
                 #M_blocks = Motion_blocks
                     
                 M_blocks = smoke_classify(
                                  frames = frames,
                                  Motion_blocks = Motion_blocks,
                                  locate_list = locate_list,
                                  WT = WT,
                                  net = net,
                                  mean_file = mean_file
                                  )
                    
                 if M_blocks:
                     if fsflag:
                         first_smoke_frame = frame_count
                         fsflag = False 
                     error_frame+=1
                         
                 for i in range(8):
                     frames.popleft()
             
             for smoke_idx in iter(M_blocks):
                 y,x,w,h = locate_list[smoke_idx]
                 #cv2.rectangle(img,(x,y),(x+24,y+24),(0,255,0),1)
                 prevpoint = (int(x+12),int(y+12))
                 cv2.circle(img, prevpoint, 15, (0,255,255))

        
             fcount = 'frame:%d'%frame_count
             cv2.putText(img,fcount,(16,16),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1,2)    
             cv2.imshow('SmokeDecetion',img)
             key = cv2.waitKey(delay)                     
             c = chr(key & 255)
             if c in [' ', 'p', 'P']:
                 if delay > 0:
                     delay = -1
                 else:
                     delay = 2
             if c in ['q','Q', chr(27)]:
                 return
         VideoIno.write("VideoPath:{} \nFirst alarm frame:{} \nError frames:{} \nAll frames:{}\n\n\n".format(VideoPath,first_smoke_frame, error_frame,frame_count))       
         VideoIno.close()         
         
         cv2.destroyWindow('SmokeDecetion')
         cv2.destroyAllWindows()
         cap.release()
    
if __name__=="__main__":   
    
    Smoke_detector('./video/1.avi')



