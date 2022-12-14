# -*- coding: utf-8 -*-
"""Untitled21.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xgXfn9CiSrxVENvj5B4EKJEzFxph76tE
"""

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip,VerticalFlip,Rotate,CLAHE,RandomBrightness

from google.colab import drive
drive.mount('/content/drive')

!mv '/content/drive/MyDrive/000_00.zip' '/content'

!unzip '/content/000_00.zip'

!pip install segmentation-models

import cv2
cv2.__version__

import glob
import matplotlib.image as mpimg
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import MeanIoU
import os
import ntpath
from google.colab.patches import cv2_imshow

images=glob.glob('/content/1_augment/img/*.png')


print(len(images))

image_color = mpimg.imread(images[0])
#image_bw = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

print(image_color.shape)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image_color)

green_channe = image_color[:,:,1]

cv2_imshow(green_channe)
green_channe = np.float32(green_channe)
green_channel = cv2.cvtColor(green_channe, cv2.COLOR_BGR2GRAY)

cv2.waitKey(0)
          
cv2.destroyAllWindows()

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(green_channel)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 

cl1 = clahe.apply(green_channel)

cv2_imshow(cl1)

cv2.waitKey(0)
          
cv2.destroyAllWindows()

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(cl1)

filename = ntpath.basename(images[0])

print(filename)

cv2.imwrite("/content/NEW/IMG/"+filename, green_channel)

for ptr in images:
    
    image_color = cv2.imread(ptr)
    
    print(image_color.shape)
    
    green_channel = image_color[:,:,1]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 

    cl1 = clahe.apply(green_channel)
    
    filename = ntpath.basename(ptr)
    
    cv2.imwrite("/content/NEW/IMG/"+filename, cl1)
    
print('Finished Saving')

all_img = glob.glob('/content/1_augment/gt/*.png')
other_dir = '/content/why/gt/'
for ptr, img_path in enumerate(all_img):
    img = cv2.imread(img_path,0)

    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    
    cv2.imwrite(f'{other_dir}/clahe_21_{img_id}.jpg',cl1)

dataset_path="/content/Disc_clahe_train"





/content/NEW/IMG

def create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def load_data(dataset_path):
  train_x=sorted(glob(os.path.join(dataset_path,"Disc/*")))
  train_y=sorted(glob(os.path.join(dataset_path,"gt/*")))
  

  

  return (train_x, train_y)

(train_x,train_y)=load_data(dataset_path)
print(f"Training:{len(train_x)}-{len(train_y)}")
#print(f"Testing:{len(test_x)}-{len(test_y)}")
#print(f"Validation:{len(val_x)}-{len(val_y)}")

create_dir("new_data/train/image")
create_dir("new_data/train/T_OD")
create_dir("new_data/test/image")
create_dir("new_data/test/T_OD")

def augment_data(Images, Train_OD, save_path, augment=True):
    size = (256,256)

    for idx, (x, y) in tqdm(enumerate(zip(Images, Train_OD)), total=len(Images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.imread(y)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, T_OD=y)
            x1 = augmented["image"]
            y1 = augmented["T_OD"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, T_OD=y)
            x2 = augmented["image"]
            y2 = augmented["T_OD"]


            aug = Rotate(limit=90,p=1.0)
            augmented = aug(image=x, T_OD=y)
            x3 = augmented["image"]
            y3 = augmented["T_OD"]


            aug = Rotate(limit=60,p=1.0)
            augmented = aug(image=x, T_OD=y)
            x4 = augmented["image"]
            y4 = augmented["T_OD"]

            aug = Rotate(limit=30,p=1.0)
            augmented = aug(image=x, T_OD=y)
            x5 = augmented["image"]
            y5 = augmented["T_OD"]

            

            X = [x, x1, x2, x3,x4,x5]
            Y = [y, y1, y2, y3,y4,y5]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            T_OD_path = os.path.join(save_path, "T_OD", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(T_OD_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

augment_data(train_y, train_x, "new_data/train/", augment=True)

augment_data(train_x,train_y,"new_data/test/",augment=True)

! zip -r 0aug.zip /content/new

from google.colab import files
files.download("/content/0aug.zip")