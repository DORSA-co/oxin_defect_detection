import DataReader as dr
import cv2
import numpy as np
import os


def bar(counter,n):
    percent = counter/n*100
    os.system('cls')
    print('percent={:.2f}%  |{}| n={} of {}'.format(percent, int(percent/2)*'◼' + int(50 - percent/2)*'-' ,int(counter), n))
    
#_________________________________________________________________________________________________________________
#explain:
#   get dataset path and build binary dataset
#
#arg:
#   src_path: main path of dataset that contain to subfolder for image and annotation
#   dst_path: main path for dst dataset
#   image_folder: image sub folder in src_path
#   annotation_folder: annotation sub folder in src_path
#
#return:
#   None
#_________________________________________________________________________________________________________________
def binary(src_path, dst_path, image_folder='image', annotation_folder='anonation'):
    #assert os.path.isdir(dst_path), 'dst_path is invalid'
    #-------------------------------------
    #define paths
    annotation_path = os.path.join(src_path, annotation_folder)
    image_path = os.path.join(src_path, image_folder)
    
    defective_path = os.path.join( dst_path, 'defective' )
    perfect_path = os.path.join( dst_path, 'perfect' )
    #-------------------------------------
    #create dst dir if not exist
    __route__ = os.path.split(dst_path)
    for i in range(len(__route__)):
        __path__ = '//'.join(__route__[:i+1])
        if not os.path.exists(__path__):
            os.mkdir(__path__)
    
    for __path__ in [defective_path,perfect_path]:
        if not os.path.exists(__path__):
            os.mkdir(__path__)
    #-------------------------------------
    #get list of annotations( name of annotations )
    annotations_name = dr.get_annotation_name(annotation_path, shuffle=False) 
    count = len(annotations_name)
    #-------------------------------------
    #Extract binary Dataset
    for i, name in enumerate(annotations_name):
        #print(nannonation)
        annotation = dr.read_annotation( annotation_path, name)
        fname = annotation.get_fname()
        img = annotation.get_img()
        if annotation.have_object() == False:
            cv2.imwrite(os.path.join(perfect_path, fname), img)
            
        else:
            cv2.imwrite(os.path.join(defective_path, fname), img)

        if i%25==0:
            bar(i+1,count)
    bar(i+1,count)



#_________________________________________________________________________________________________________________
#explain:
#   get dataset path and build mask dataset
#
#arg:
#   src_path: main path of dataset that contain to subfolder for image and annotation
#   dst_path: main path for dst dataset
#   num_classes: number of classes
#   mask_size: size of mask for save (h,w)
#   image_folder: image sub folder in src_path
#   annotation_folder: annotation sub folder in src_path
#   multi_mask: if be True mask of each class save in seprate forlder another way mask of all class add to each other and save in one image
#   consider_no_object= if True, it Allocates a new class to background. it's class is 0 class. defuat is False
#   class_id: it can be an integer that show id of class and just extract mask of that class
#return:
#   None
#_________________________________________________________________________________________________________________
def mask(src_path,
         dst_path,
         num_classes,
         mask_size,
         image_folder='image',
         annotation_folder='anonation',
         multi_mask=False,
         consider_no_object=False,
         class_id=None
         ):
    if not multi_mask :
        assert not consider_no_object, 'consider_no_object enabeld but multi_mask is False'
    #-------------------------------------
    #define paths
    annotation_path = os.path.join(src_path, annotation_folder)
    image_path = os.path.join(src_path, image_folder)
    
    image_path = os.path.join( dst_path, 'image' )
    label_path = os.path.join( dst_path, 'label' )
    #-------------------------------------
    #create dst dir if not exist
    __route__ = os.path.split(dst_path)
    for i in range(len(__route__)):
        __path__ = '//'.join(__route__[:i+1])
        if not os.path.exists(__path__):
            os.mkdir(__path__)
    
    for __path__ in [image_path,label_path]:
        if not os.path.exists(__path__):
            os.mkdir(__path__)
    
    if multi_mask:
        for i in range(num_classes):
            __path__ = os.path.join( label_path,str(i+1))
            if not os.path.exists(__path__):
                os.mkdir(__path__)
        
        if consider_no_object:
            __path__ = os.path.join( label_path,'0')
            if not os.path.exists(__path__):
                os.mkdir(__path__)
    #-------------------------------------
    func = dr.extract_mask(num_classes, mask_size, consider_no_object=consider_no_object, class_id=class_id)
    #-------------------------------------
    #get list of annotations( name of annotations )
    annotations_name = dr.get_annotation_name(annotation_path, shuffle=False) 
    count = len(annotations_name)
    #-------------------------------------
    #Extract binary Dataset
    for i, name in enumerate(annotations_name):
        #print(nannonation)
        annotation = dr.read_annotation( annotation_path, name)
        fname = annotation.get_fname()
        if not annotation.have_object():
            continue
        img, msks  = func(annotation)
        
        if not multi_mask:
            res_mask = np.sum(msks.astype(np.int32), axis=-1)
            res_mask = np.clip(res_mask,0,255).astype(np.uint8)

            cv2.imwrite(os.path.join(image_path, fname), img)
            cv2.imwrite(os.path.join(label_path, fname), res_mask)
        
        else:
            msks = np.moveaxis(msks, -1,0)
            cv2.imwrite(os.path.join(image_path, fname), img)
            for _class_,res_mask in enumerate(msks):
                if consider_no_object: #class should save from zero
                    cv2.imwrite(os.path.join(label_path,str(_class_), fname), res_mask)
                    
                else: #class should save from one
                    cv2.imwrite(os.path.join(label_path,str(_class_+1), fname), res_mask)

        if i%25==0:
            bar(i+1,count)
    bar(i+1,count)


  
if __name__ == '__main__':
    
    #binary('severstal-steel-defect-detection/', 'data/allb', 'train_images', 'annotations')
    
    mask(src_path='severstal-steel-defect-detection/',
         dst_path='data/allm',
         num_classes=4,
         mask_size=(256,1600),
         image_folder='train_images',
         annotation_folder='annotations',
         multi_mask=False,
         consider_no_object=False
         )
    
         