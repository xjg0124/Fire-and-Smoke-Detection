# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:12:23 2016

@author: xjg
"""
import numpy as np
import cv2
import cv2.cv as cv

class Smoke_tool():
      
    def __init__(self):
        self.name = '' 

    #图片分块，WD为块大小  
    def ImageBlock(self,rows,cols,WD):
        result = list()
        for i in range(0, rows-WD, WD):
            for j in range(0, cols, WD):
                box = (j, i, WD, WD)        
                result.append(box)
        return result
        
    #分块运动检测    
    def MotionDetection(self,gray_img,img_back_gray,locate_list,WD):
        
        T = WD*WD/2
        #T = 30
        TH = 4
        result = list()
        
        gray_back = np.zeros(gray_img.shape, np.uint8) 
        gray_cur = np.zeros(gray_img.shape, np.uint8) 

        gray_back = img_back_gray   
        gray_cur = gray_img
        
        diff = cv2.absdiff(gray_back, gray_cur)
        diff /= TH
        
        int_diff = cv2.integral(diff,-1)            
        idx = 0            
        for pt in iter(locate_list):
            xx,yy,d1,d2 = pt
            t11 = int_diff[xx,yy]
            t22 = int_diff[xx+WD,yy+WD]
            t12 = int_diff[xx,yy+WD]
            t21 = int_diff[xx+WD,yy]
            block_diff = t11 + t22 - t12 - t21
            if block_diff > T:
                result.append(idx)
            idx+=1
        return result
        
    def ColorDetection(self,img,locate_list,WD):
        
        alpha = 20
        L1 = 150
        L2 = 220
        D1 = 80
        D2 = 150
        length_c = 0
        result = list()
        
        
        idx = 0
        for pt in iter(locate_list):
            yy,xx,w,h = pt

            for i in range(yy,yy+w):
                for j in range(xx,xx+h):
                    if (abs(img[i,j,0]-img[i,j,1])<alpha and abs(img[i,j,0]-img[i,j,2])<alpha and abs(img[i,j,1]-img[i,j,2])<alpha):
                        r1 = True
                    else:
                        r1 = False
                    I = (max(img[i,j,0],img[i,j,1],img[i,j,2]) + min(img[i,j,0],img[i,j,1],img[i,j,2])) / 2
                    if((L1<=I) and (I<=L2)):
                        r2 = True
                    else:
                        r2 = False
                    if((D1<=I) and (I<=D2)):
                        r3 = True
                    else:
                        r3 = False
                    if(r1 and (r2 or r3)):
                        length_c+=1

            if length_c > WD*WD*0.6:
                result.append(idx)
            idx+=1
        return result
    def OpticalFlowPara(self,img):
        width = img.shape[0]
        height = img.shape[1]
 
        self.prevPyr = cv.CreateImage((height / 3, width + 8), 8, cv.CV_8UC1) #Will hold the pyr frame at t-1
        self.currPyr = cv.CreateImage((height / 3, width + 8), 8, cv.CV_8UC1) # idem at t
 
        self.max_count = 500
        self.qLevel= 0.001
        self.minDist = 10
        self.prev_points = [] #Points at t-1
        self.curr_points = [] #Points at t
        
        
    def MotionDetectionOpticalFlow(self,gray,back_gray,WD):
        
        
        result=[] #To keep all the lines overtime
        mat_gray = cv.fromarray(gray)
        mat_prev_gray = cv.fromarray(back_gray)
 
        self.prev_points = cv.GoodFeaturesToTrack(mat_gray, None, None, self.max_count, self.qLevel, self.minDist) #Find points on the image
        #Calculate the movement using the previous and the current frame using the previous points
        self.curr_points, status, err = cv.CalcOpticalFlowPyrLK(mat_prev_gray, mat_gray, self.prevPyr, self.currPyr, self.prev_points, (10, 10), 3, (cv.CV_TERMCRIT_ITER|cv.CV_TERMCRIT_EPS,20, 0.03), 0)
 
 
        #If points status are ok and distance not negligible keep the point
        k = 0
        for i in range(len(self.curr_points)):
            nb =  abs( int(self.prev_points[i][0])-int(self.curr_points[i][0]) ) + abs( int(self.prev_points[i][1])-int(self.curr_points[i][1]) )
            if status[i] and  nb > 2 :
                self.prev_points[k] = self.prev_points[i]
                self.curr_points[k] = self.curr_points[i]
                k += 1
 
        self.prev_points = self.prev_points[:k]
        self.curr_points = self.curr_points[:k]
        #At the end only interesting points are kept
 
        #Draw all the previously kept lines otherwise they would be lost the next frame
        #for (pt1, pt2) in lines:
        #   cv2.line(frame, pt1, pt2, (255,255,255))
 
        #Draw the lines between each points at t-1 and t
        for prevpoint, point in zip(self.prev_points,self.curr_points):
            prevpoint = (int(prevpoint[0]),int(prevpoint[1]))
            #cv2.circle(img, prevpoint, 15, 0)
            point = (int(point[0]),int(point[1]))

            n = int(prevpoint[0] / 24)*int(gray.shape[0] / 24) + int(prevpoint[1] / 24)

            if n < int(gray.shape[0] / 24)*int(gray.shape[1] / 24):
                result.append(n)
 
        self.prev_points = self.curr_points
        
        result = list(set(result))
        return result
        
    
        