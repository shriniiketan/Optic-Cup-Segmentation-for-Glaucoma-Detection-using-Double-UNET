import glob
import matplotlib.image as mpimg
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
import os
import ntpath

disc=glob.glob('C:/Optic Cup Segmentation/areal_cup/masks_disc/*')
cup=glob.glob('C:/Optic Cup Segmentation/areal_cup/masks_cup/*')
images=glob.glob('C:/Optic Cup Segmentation/areal_cup/images/*')


print(len(disc))
'''
image_color = cv2.imread(disc[1])

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image_color)

image_gray = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)

high = 0

low = 10000000000 

big = 0

small = 1000000000

cord1 = 0
cord2 = 0
cord3 = 0
cord4= 0

for i in range (image_gray.shape[0]):
   for j in range (image_gray.shape[1]):
       
       if image_gray[i,j] == 255:
           
           if i > high:
               high = i
               cord1 = (high,j)
               
           if i < low:
               low = i
               cord2 = (low,j)
               
           if j > big:
               big = j
               cord3 = (i,big)
             
           if j < small:
               small = j
               cord4 = (i,small)

print(cord1)  
print(cord2) 
print(cord3)  
print(cord4) 
             
a = cord1[0] 
b = cord1[1] 
c = cord2[0] 
d = cord2[1]

e = cord3[0] 
f = cord3[1] 
g = cord4[0] 
h = cord4[1]


print(a)
print(b)
print(c)
print(d)

print(e)
print(f)
print(g)
print(h)

cv2.line(image_color, (b,a), (d,c),(255,0,0), 10)
cv2.line(image_color, (f,e), (h,g),(255,0,0), 10)


plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image_color)

crop = image_color[c-20:a+20, h-20:f+20]

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(crop)
'''
count1 = 0
count2 = 0
for ptr in disc:
    
    high = 0
    low = 10000000000
    big = 0
    small = 1000000000000
    
    
    image_color = cv2.imread(ptr)
    
    image_gray = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)
    
    ori_image = cv2.imread(images[count1])
    
    ori_cup = cv2.imread(cup[count2])
    
    print(image_gray.shape)
    
    for i in range (image_gray.shape[0]):
       for j in range (image_gray.shape[1]):
           
           if image_gray[i,j] == 255:
               
               if i > high:
                   high = i
                   cord1 = (high,j)
                   
               if i < low:
                   low = i
                   cord2 = (low,j)
                   
               if j > big:
                   big = j
                   cord3 = (i,big)
                 
               if j < small:
                   small = j
                   cord4 = (i,small)
    '''
    
    if cord2[0] >= 20:
        a = cord2[0] - 20 
    else :
        a = cord2[0] 
    
    
    if cord4[1] >= 20:
        b = cord4[1] - 20 
    
    else :
        b = cord4[1] 
    
    '''
    
    crop_disc = image_gray[cord2[0]:cord1[0], cord4[1]:cord3[1]]
    
    crop_cup = ori_cup[cord2[0]:cord1[0], cord4[1]:cord3[1]]
    
    crop_image = ori_image[cord2[0]:cord1[0], cord4[1]:cord3[1]]

              
    filename1 = ntpath.basename(ptr)
    
    filename2 = ntpath.basename(images[count1])
    
    filename3 =ntpath.basename(cup[count2])
    
    cv2.imwrite("C:/Optic Cup Segmentation/Cropped_Data_new_CLAHE/crp_disc/"+filename1, crop_disc)
    
    cv2.imwrite("C:/Optic Cup Segmentation/Cropped_Data_new_CLAHE/crp_cup/"+filename3, crop_cup)
    
    cv2.imwrite("C:/Optic Cup Segmentation/Cropped_Data_new_CLAHE/crp_img/"+filename2, crop_image)
    
    count1 = count1+1
    count2 = count2+1

print('Saving Over')








            