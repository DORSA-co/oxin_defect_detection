from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import copy

aug_dict = dict(rotation_range=15,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.05,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='constant',            
                    )


#_________________________________________________________________________________________________________________
#explain:
#  return two binary generator (train and test)
#
#arg:
#   path: path of dataset
#   target_size: target size of image
#   num_classes: number of classes
#   defective_folder: subfolder for defective image
#   perfect_folder: subfolder for perfect image
#   batch_size: size of batch
#   validation_split = validation split percent 
#
#return:
#   trainGen, testGen
#
#_________________________________________________________________________________________________________________
def get_binarygenerator(path, target_size, defective_folder, perfect_folder,aug_dict , batch_size=8, validation_split=0.2):
    
    trainGen = ImageDataGenerator(**aug_dict, validation_split=validation_split)
    valGen = ImageDataGenerator(validation_split=validation_split)
    
    
    train_dataset = trainGen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=[perfect_folder, defective_folder],
        seed=42,
        subset='training') # set as training data

    val_dataset = trainGen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=[perfect_folder, defective_folder],
        seed=42,
        subset='validation') # set as training data

    return train_dataset, val_dataset











#_________________________________________________________________________________________________________________
#explain:
#  Generator for mask
#
#arg:
#   path: path of dataset
#   image_folder: subfolder for images
#   mask_folder: subfolder for masks
#   aug_dict: augmention dictionary for ImageDataGenerator
#   subfolders_mask: if mask_folder contain Some folder for each mask's class it should be a list of class's subfolder ( order is import )
#                       another way it should be None ( for one class mask )
#   target_size: target size of image
#   .
#   .
#   .
#
#yield:
#   batch_imgs, batch_masks
#   batch_imgs: batch of image
#   batch_masks: batch of masks
#
#_________________________________________________________________________________________________________________
def maskGenerator(path,
                  image_folder,
                  mask_folder,
                  aug_dict,
                  subfolders_mask=None,
                  target_size = (256,256),
                  batch_size=8,
                  seed = 1,
                  image_color_mode = "grayscale",
                  mask_color_mode = "grayscale",
                  image_save_prefix  = "image",
                  mask_save_prefix  = "mask",
                  bg_idx=None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    #-----------------------------------------------------
    def adjustData(img,mask):
        if(np.max(img) > 1):
            img = img / 255.
            mask = mask /255.
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            return (img,mask)
    #-----------------------------------------------------
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    #-----------------------------------------------------
    image_generator = image_datagen.flow_from_directory(
        path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_prefix  = image_save_prefix,
        seed = seed)
    #-----------------------------------------------------
    mask_generator = []
    if subfolders_mask is None:
        mask_generator = mask_datagen.flow_from_directory(
            path,
            classes = [mask_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size, 
            batch_size = batch_size ,
            save_prefix  = mask_save_prefix,
            seed = seed)
    
    else:
        for subfolder in subfolders_mask:
            
            if bg_idx is not None:
                if subfolder == subfolders_mask[bg_idx]:
                    mask_datagen.cval = 255
                else:
                    mask_datagen.cval = 0
                    
            mask_generator.append(
                copy.deepcopy(mask_datagen).flow_from_directory(
                os.path.join(path, mask_folder),
                classes = [subfolder] ,
                class_mode = None,
                color_mode = mask_color_mode,
                target_size = target_size, 
                batch_size = batch_size ,
                save_prefix  = mask_save_prefix,
                seed = seed)
                )
    #-----------------------------------------------------
    if subfolders_mask is None:
        train_generator = zip( image_generator , mask_generator )
        for (img,mask) in train_generator:
            img,mask = adjustData(img,mask)
            yield (img,mask)
    
    else:
        mask_generator = tuple(mask_generator)
        train_generator = zip( *((image_generator,) + mask_generator) )
        for (img,*mask) in train_generator:
            masks = np.concatenate(mask, axis=-1)
            img,masks = adjustData(img,masks)
            yield (img,masks)
    #-----------------------------------------------------


if __name__=='__main__':
    train, test = get_binarygenerator('data//binary',
                                      (128,800),
                                      defective_folder='defective',
                                      perfect_folder='perfect',
                                      aug_dict=aug_dict,
                                      validation_split=0.2)

    train_c = maskGenerator('data/mask_class/train',
                          'image',
                          'label',
                          aug_dict, 
                          subfolders_mask=['0','1','2','3','4'], 
                          batch_size=8,
                          target_size=(128,800),
                          bg_idx=0)    
    
    
    train_nc = maskGenerator('data/mask/train',
                          'image',
                          'label',
                          aug_dict, 
                          batch_size=1,
                          target_size=(128,800))
    
    import cv2
    for x,ys in train_c:
        x = x[0]
        ys=ys[0]
        x = x * 255
        ys = ys * 255
        x = x.astype(np.uint8)
        ys = ys.astype(np.uint8)
        
        
        cv2.imshow('x',x)
        ys = np.moveaxis(ys,[-1],[0])
        for i, y in enumerate(ys):
            cv2.imshow('y{}'.format(i),y)
        
        cv2.waitKey(0)