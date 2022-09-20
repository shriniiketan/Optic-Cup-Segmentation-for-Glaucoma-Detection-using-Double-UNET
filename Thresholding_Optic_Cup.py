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

images=glob.glob('C:/Optic Cup Segmentation/Opticup_Dataset/CUP images/r_optic_cup/*.bmp')

print(len(images))

image_color = cv2.imread(images[0])

print(image_color.shape)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image_color)

print(image_color.shape)

image_gray = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)

print(image_gray.shape)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image_gray)

cv2.imshow("GRAY", image_gray)

cv2.waitKey(0)
          
cv2.destroyAllWindows()

print(np.unique(image_gray))

print(image_gray.shape[1])

for i in range (image_gray.shape[0]):
   for j in range (image_gray.shape[1]):
       if image_gray[i,j] < 255 and image_gray[i,j] > 0:
           image_gray[i,j] = 255

cv2.imshow("GRAY", image_gray)

cv2.waitKey(0)
          
cv2.destroyAllWindows()


for ptr in images:
    
    image_color = cv2.imread(ptr)
    
    image_gray = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)
    
    print(image_gray.shape)
    
    for i in range (image_gray.shape[0]):
        for j in range (image_gray.shape[1]):
            if image_gray[i,j] < 255 and image_gray[i,j] > 0:
                image_gray[i,j] = 0
            elif image_gray[i,j] == 0:
                image_gray[i,j] = 255
            elif image_gray[i,j] == 255:
                image_gray[i,j] = 0
              
    filename = ntpath.basename(ptr)
    
    cv2.imwrite("C:/Optic Cup Segmentation/r_optic_cup_correct/"+filename, image_gray)
    
print('Saving Over')