import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from sklearn import linear_model

class find_rotation(object):
    
    def __init__ (self):
        im=[]
    
    def find_yaw(img, simulator=False):
        if simulator :
            img2gray = img
        else:
            img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        unique, counts = np.unique(img2gray, return_counts=True)     
        yaw=0
        cX=cY=0
        if len(unique) > 0:
             for a in unique[1:]:
                 b = img2gray==a                                                             # create a boolian mask for each instance
                 data = [[int(x) for x in y] for y in b]                                     # convert boolian array to int. array
                 h = np.array(data)
                 i=np.uint8(h)                                                         
                 # Find contours and sort them from big to small
                 contour, _ = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                 cnts = sorted(contour, key=cv2.contourArea, reverse=True)
                 cnt = cnts[0]
                 if not cv2.isContourConvex(cnt):
                     # Find center of contour area
                     M = cv2.moments(cnt) 
                     if  M["m00"] !=0 :
                         cX = int(M["m10"]/ M["m00"] )
                         cY = int(M["m01"]/ M["m00"] )
    
                         if img2gray[cY,cX] != 0:
                        
                             rows,cols = img.shape[:2]
                             [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
                             lefty = int((-x*vy/vx) + y)
                             righty = int(((cols-x)*vy/vx)+y)
                             cv2.line(img2gray,(cols-1,righty),(0,lefty),(0,255,0),2)
                             yaw = math.atan(-vy/vx)
                 else:
                    print ("Not Convex")   
        else:
            print ("Nothing found")
    
        return yaw , img2gray , [cX,cY]
