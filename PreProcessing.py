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

images=glob.glob('C:/Optic Cup Segmentation/image/*.jpg')

print(len(images))

image_color = mpimg.imread(images[0])

print(image_color.shape)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image_color)

green_channel = image_color[:,:,1]

cv2.imshow("CLAHE", green_channel)

cv2.waitKey(0)
          
cv2.destroyAllWindows()

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(green_channel)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 

cl1 = clahe.apply(green_channel)

cv2.imshow("CLAHE", cl1)

cv2.waitKey(0)
          
cv2.destroyAllWindows()

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(cl1)

filename = ntpath.basename(images[0])

print(filename)

cv2.imwrite("C:/Optic Cup Segmentation/"+filename, green_channel)

for ptr in images:
    
    image_color = cv2.imread(ptr)
    
    print(image_color.shape)
    
    green_channel = image_color[:,:,1]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 

    cl1 = clahe.apply(green_channel)
    
    filename = ntpath.basename(ptr)
    
    cv2.imwrite("C:/Optic Cup Segmentation/New_Data/image/"+filename, cl1)
    
print('Finished Saving')