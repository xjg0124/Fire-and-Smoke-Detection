'''
A sample script to run c3d classifications on multiple videos
'''

import os
import numpy as np
import cv2
import math
import json
import sys
sys.path.append("../python")
import caffe
from c3d_classify import c3d_classify
from collections import deque


def smoke_classify(
        frames,
        Motion_blocks,
        locate_list,
        WT,
        net,
        mean_file
        ):



    # network param
    prob_layer = 'prob'
    result = list()
    
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file,'rb').read()
    blob.ParseFromString(data)
    image_mean = np.array(caffe.io.blobproto_to_array(blob))
    
    #idx = 0
    for smoke_idx in iter(Motion_blocks):
        yy,xx,w,h = locate_list[smoke_idx]
    #for pt in iter(locate_list):

       
        
        img_blocks = deque()
        #yy,xx,w,h = pt
        
        
        for i in range(WT):
            img = frames[i]
            block = img[yy:yy+h,xx:xx+w]
            img_blocks.append(img[yy:yy+h,xx:xx+w])
          
        
        
        
        prediction = c3d_classify(
            img_blocks=img_blocks,
            image_mean=image_mean,
            net=net,
            prob_layer=prob_layer,
            multi_crop=False
            )
        
        
        
        if prediction.ndim == 2:
            avg_pred = np.mean(prediction, axis=1)
        else:
            avg_pred = prediction
            

        if avg_pred[1] >= 0.99:
            result.append(smoke_idx)
            #result.append(idx)


        #idx+=1
        img_blocks.clear()
        
        
        

        
    return result
    
