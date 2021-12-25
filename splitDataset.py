import os
import random
import sys
import shutil
import numpy as np

src_path = 'data/steel_defects/all/'
dst_path = 'data/steel_defects'
split = 0.2

def bar(counter,n):
    percent = counter/n*100
    os.system('cls')
    print('percent={:.2f}%  |{}| n={} of {}'.format(percent, int(percent/2)*'◼' + int(50 - percent/2)*'-' ,int(counter), n))
    
    
    
#_________________________________________________________________________________________________________________
#explain:
#   split mask dataset into train and test
#
#arg:
#   src_path: main path of dataset that contain to subfolder for image and annotation
#   dst_path: main path for dst dataset
#   image_folder: image sub folder in src_path
#   label_folder: label sub folder in src_path
#   multi_mask_class: if label_folder contain some folder for each class it should be True, O.W if label folder 
#                       only contain images for one class it should be False
#   split: split percent for Test                        
#return:
#   None
#_________________________________________________________________________________________________________________
def split_unet_dataset(src_path, dst_path, img_folder='image', label_folder='label', mulit_mask_class=False, split=0.2):
    imgs_list = os.listdir( os.path.join( src_path, img_folder))
    #-------------------------------------
    #calc train and test count for split
    nimage = len(imgs_list)
    ntest = int(split * nimage)
    ntrain= nimage - ntest
    print("{} images split into {} train / {} test".format(nimage, ntrain, ntest))
    #-------------------------------------
    #obtain sources path for image and mask
    images_dir = os.path.join(src_path, img_folder)
    labels_dir = os.path.join(src_path, label_folder)
    #-------------------------------------
    folders_1 = ['train', 'test']
    folders_2 = ['image', 'label']
    if mulit_mask_class:
        folders_classes = os.listdir(labels_dir)
        folders_classes = list( filter ( lambda x:os.path.isdir(os.path.join(labels_dir,x)),
                                     folders_classes ))
        
        assert len(folders_classes)>0, 'multi_classes is True but Threre is no sub folder in {}'.format(os.path.join(src_path,label_folder))

            
    #-------------------------------------
    #remove dst contain folders if exist
    for f1 in folders_1:
        _path_ = os.path.join(dst_path,f1)
        if os.path.exists( _path_ ):
            shutil.rmtree( _path_ )
            
    
            
    #-------------------------------------
    __route__ = os.path.split(dst_path)
    for i in range(len(__route__)):
        __path__ = '//'.join(__route__[:i+1])
        if not os.path.exists(__path__):
            os.mkdir(__path__)
    #-------------------------------------
    
    #create all sub folders for dst
    for f1 in folders_1:
        _path_ = os.path.join( dst_path, f1)
        os.mkdir(_path_)
    
    for f1 in folders_1:
        for f2 in folders_2:
            _path_ = os.path.join( dst_path, f1,f2)
            os.mkdir(_path_)
    
    if mulit_mask_class:
        for f1 in folders_1:
            for f3 in folders_classes:  
                _path_ = os.path.join( dst_path, f1,folders_2[1],f3 )
                os.mkdir(_path_)

    #-------------------------------------
    #split images list into train and test
    random.shuffle(imgs_list)
    test_img_list = imgs_list[:ntest]
    train_img_list = imgs_list[ntest:]  
    lists_dict = {folders_1[0]:train_img_list, folders_1[1]:test_img_list}
    #-------------------------------------------------
    counter = 0
    for f1,fname_list in lists_dict.items():    
        for fname in fname_list:
            for f2 in folders_2:
                
                if f2 == 'image': #just copy image into corespond label
                    src = os.path.join(src_path, img_folder, fname )
                    dst = os.path.join(dst_path,f1, f2, fname )
                    shutil.copy( src, dst)
                
                else:
                    if not mulit_mask_class:
                        src = os.path.join(src_path, label_folder, fname )
                        dst = os.path.join(dst_path,f1, f2, fname )
                        shutil.copy( src, dst)
                    
                    else:
                        for f3 in folders_classes:
                            src = os.path.join(src_path, label_folder,f3, fname )
                            dst = os.path.join(dst_path,f1, f2, f3, fname )
                            shutil.copy( src, dst)
                            
                        
            counter+=1  
            if counter%5==0:
                bar(counter,nimage)
    bar(counter,nimage)











if __name__ == '__main__':
    split_unet_dataset('data/allmc', 'data/mask_class', img_folder='image', label_folder='label', mulit_mask_class=True, split=0.2)


