import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator






train_dir_img = ''
train_dir_mask = ''
validation_dir_img = ''
validation_dir_mask = ''



test_datagen = ImageDataGenerator(rotation_range = 60,
                                   width_shift_range = 0.2,
                                   height_shift_range =0.2,
                                   shear_range =0.2,
                                   harizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = 'nearest')

def trainGenerator(train_dir_img, train_dir_mask):
    
    train_datagen = ImageDataGenerator(rotation_range = 60,
                                   width_shift_range = 0.2,
                                   height_shift_range =0.2,
                                   shear_range =0.2,
                                   harizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = 'nearest')
    
    train_generator_image = train_datagen.flow_from_directory(
            train_dir_img,  
            batch_size=20,        
            class_mode='binary')
    
    train_generator_mask = train_datagen.flow_from_directory(
            train_dir_mask,  
            batch_size=20,        
            class_mode='binary')
    
    train_generator = zip(train_generator_image, train_generator_mask)

train_img_gen = trainGenerator(train_dir_img, train_dir_mask)

val_img_gen = trainGenerator(validation_dir_img, validation_dir_mask)

history = model.fit(
      train_img_gen,
      steps_per_epoch=100,  
      epochs=100,
      validation_data=val_img_gen,
      validation_steps=50,  
      verbose=2)

