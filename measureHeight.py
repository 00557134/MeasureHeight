# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:17:39 2021

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('pic1.jpg') 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度影象 


face_cascade = cv2.CascadeClassifier('C:\\Users\\User\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

head_coord = []
foot_coord = []
horizon = []


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return round(x), round(y)
    else:
        return False
    

def faceDetect(padding_img, head_coord, foot_coord, person1_X, flag=1):
    faces = face_cascade.detectMultiScale(padding_img,
                                      scaleFactor=1.2,
                                      minNeighbors=3,)
    height = []
    if flag == 1:
        height.append(590)
    elif flag == 2:
        height.append(400)
        height.append(590)
    elif flag == 3:
        height.append(465)
        height.append(330)
    elif flag == 4:
        height.append(580)
    
    
    for i, (x,y,w,h) in enumerate(faces):
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        tmp = 0
        tmp2 = 0
        if flag == 1 and i == 0:
            tmp = -40
        elif i == 1 and flag == 2:
            tmp = -40
        elif i == 1 and flag == 3:
            tmp = -20
        elif flag == 4 and i == 0:
            tmp = -40
        
        if i == 0 and flag == 3:
            tmp2 = -30
        else:
            tmp2 = 0
            
        if i == 0 and flag == 2:
            tmp2 = -20
        else:
            tmp2 = 0
        
        padding_img = cv2.circle(padding_img, (int(x+w/2), int(y+tmp+tmp2+h/2)), radius=20, color=(255, 0, 0), thickness=-1)
        padding_img = cv2.circle(padding_img, (int(x+w/2), int(y+height[i]+h/2)), radius=20, color=(255, 255, 0), thickness=-1)
        #print(x+w/2, y+h/2)
        head_coord.append((x+w/2, y+tmp2+tmp+h/2))
        foot_coord.append((int(x+w/2), int(y+height[i]+h/2)))
        
        if i == 1 and flag != 1 and flag != 3:
            #print('Up')
            person1_X.append(int(x+w/2))
        if flag == 1 or flag == 4 or flag == 3:
            #print('Down')
            person1_X.append(int(x+w/2))
    
    #print(head_coord)
    return x, y, w, h
    

def height_cal(t, r, foot, flag=1):
    if flag == 3:
        H = (r-foot)*180/(t-foot)
    else:
        H = (t-foot)*180/(r-foot)
    
    return H


filename = 'C:\\Users\\User\\Desktop\\CV_Hw\\CV_HM4\\'
img_name = input('Image Name: ')
filename = filename + 'pic' + img_name + '.jpg' 
#flag = int(filename[-5])
flag = int(img_name)
print('Flag: ', flag)

img = cv2.imread(filename)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

WHITE = [255,255,255]
constant = cv2.copyMakeBorder(img,0,0,1000,1000,cv2.BORDER_CONSTANT,value=WHITE)

faces = face_cascade.detectMultiScale(constant,
                                      scaleFactor=1.2,
                                      minNeighbors=3,)
    
person1_X = []


x, y, w, h = faceDetect(constant, head_coord, foot_coord, person1_X, flag)

# Another Head Coordinate
if len(head_coord) != 2 and flag == 1:
    #print('Another Person')
    head_coord.append((x-410+w/2, y-35+h/2))
    foot_coord.append((int(x-410+w/2), int(y+415+h/2)))
    constant = cv2.circle(constant, (int(x-410+w/2), int(y-35+h/2)), radius=20, color=(255, 0, 0), thickness=-1)
    constant = cv2.circle(constant, (int(x-410+w/2), int(y+415+h/2)), radius=20, color=(255, 255, 0), thickness=-1)
elif len(head_coord) != 2 and flag == 4:
    #print('Another Person')
    head_coord.append((x-310+w/2, y+10+h/2))
    foot_coord.append((int(x-310+w/2), int(y+300+h/2)))
    constant = cv2.circle(constant, (int(x-310+w/2), int(y+10+h/2)), radius=20, color=(255, 0, 0), thickness=-1)
    constant = cv2.circle(constant, (int(x-310+w/2), int(y+300+h/2)), radius=20, color=(255, 255, 0), thickness=-1)

# Vanishing Line
horizon.append((int(x-500+w/2), int(y+h/2)))
horizon.append((int(x+400+w/2), int(y+h/2)))

constant = cv2.circle(constant, horizon[0][:2], radius=20, color=(0, 255, 0), thickness=-1)
constant = cv2.circle(constant, horizon[1][:2], radius=20, color=(0, 255, 0), thickness=-1)



L1 = line(foot_coord[0], foot_coord[1])
L2 = line(horizon[0], horizon[1])

R = intersection(L1, L2)

print('Intersection Point: ', R)

constant = cv2.circle(constant, R, radius=10, color=(255, 0, 255), thickness=-1)

tmp = 0


if flag == 3 or flag == 4 or flag == 1:
    tmp = 1

#print(head_coord)
#print(head_coord[tmp])
H_abc = line(R, head_coord[tmp])
#print(person1_X[0])
#print(H_abc)
project_H = int((-H_abc[0]*person1_X[0]+H_abc[2])/H_abc[1])

#print(person1_X[0], project_H)
constant = cv2.circle(constant, (person1_X[0], project_H), radius=10, color=(255, 0, 255), thickness=-1)

cv2.imwrite('./tzuyu_face.jpg', constant)


if flag != 1 and flag != 4:
    tmp = 1

if flag == 1:
    tmp = 0

#print(head_coord[tmp][1], foot_coord[tmp][1])
H = height_cal(project_H, head_coord[tmp][1], foot_coord[tmp][1], flag)
print('Height: ', H)

plt.subplot(121),plt.imshow(img)
plt.xticks([]),plt.yticks([])

plt.subplot(122),plt.imshow(constant)
plt.xticks([]),plt.yticks([])












