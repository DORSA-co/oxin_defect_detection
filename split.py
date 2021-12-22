import os
import random
import sys
import shutil
import numpy as np

src_path = 'data/steel_defects/all/'
dst_path = 'data/steel_defects'
split = 0.2


imgs_list = os.listdir( os.path.join( src_path, 'image'))
#-------------------------------------
n = len(imgs_list)
ntest = int(split * n)
ntrain= n - ntest
print("{} images split into {} train / {} test".format(n, ntrain, ntest))
#-------------------------------------
all_dir_image = os.path.join(src_path, 'image')
all_dir_label = os.path.join(src_path, 'label')

train_dir = os.path.join( dst_path, 'train')
test_dir = os.path.join( dst_path, 'test')

train_img_dir = os.path.join(train_dir,'image')
train_label_dir = os.path.join(train_dir,'label')

test_img_dir = os.path.join(test_dir,'image')
test_label_dir = os.path.join(test_dir,'label')

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

paths = [train_dir,
         train_img_dir,
         train_label_dir,
         test_dir,
         test_img_dir,
         test_label_dir]

for path in paths:
    os.mkdir(path)


random.shuffle(imgs_list)
test_list = imgs_list[:ntest]
train_list = imgs_list[ntest:]

all_dict = {'train':train_list, 'test':test_list}
subfolder_list = ['image', 'label']
#-------------------------------------------------
complelet = n * 2
counter = 0
pre_percent = -1
for folder,fname_list in all_dict.items():    
    for fname in fname_list:
        for subfolder in subfolder_list:
            src = os.path.join(src_path, subfolder, fname )
            dst = os.path.join(dst_path,folder, subfolder, fname )

            shutil.copy( src, dst)

            counter+=1
            percent = counter/complelet * 100
            if counter%20==0:
                os.system('cls')
                print('percent={:.2f}%  |{}| n={} of {}'.format(percent, int(percent/2)*'#' + int(100 - percent/5)*'-' ,int(counter/2), n))
            
            


os.system('cls')
print('percent={:.2f}%  |{}| n={}'.format(100, 50*'#' ,len(imgs_list)))
    
