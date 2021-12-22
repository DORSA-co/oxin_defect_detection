import csv
import os
import numpy as np
import cv2
from os import path
import json
#______________________________________________________________________________________________________________________________________________
#expalin:
#   read csv file from path and return csv list
#
#arg:
#   csv_path: path of csv label
#
#return:
#   csv list: 
#       col_0: image_name, col_1: Class_ID, col_2: Encoded_pixel
#______________________________________________________________________________________________________________________________________________
def csv_reader( csv_path ):
    with open( csv_path, newline='') as csvfile :
        csv_iter = csv.reader( csvfile)
        csv_file = list(csv_iter)
        return csv_file[1:]  

#______________________________________________________________________________________________________________________________________________
#explain:
#   ->Convert csv list to dictionay that each key is a image_name and value is a tupel contatin classes_list and Encodded_pixel_list
#   ->'image_name': ([classes], [encoded_pilxes_array])
#   ->class: an int number
#   ->encoded_pilxes_array: an array with shape(n,2) that first col is start pixel and second col is count pix ( order is form top to down then lft to right)
#
#arg:
#   csv_list: output of csv_reader
#
#return:
#   dict label
#______________________________________________________________________________________________________________________________________________
def csv2labelDict( csv_list ):
    dict_lbl = {}
    #Row -> image_name, class_id, mask_row_lenght_code
    for row in csv_list:
        img_name, class_id, encoded_pixel = row
        class_id = int(class_id) - 1 #in csv class start from 1
        encoded_pixel = encoded_pixel.split(' ')
        encoded_pixel = list(  map( int, encoded_pixel ))
        encoded_pixel = np.array(encoded_pixel).reshape( (-1,2) ) #firrst col-> start pix , second col -> count pix

        if dict_lbl.get( img_name ) is None:
            dict_lbl[ img_name ] = ( [class_id], [encoded_pixel] )
        
        else:
            classes_list, encodedPx_list = dict_lbl.get( img_name )
            classes_list.append( class_id )
            encodedPx_list.append( encoded_pixel )
            dict_lbl[img_name] = (classes_list, encodedPx_list)
    return dict_lbl


#______________________________________________________________________________________________________________________________________________
#explain:
#   ->A function which creates a progress bar in console, full documentation in the link bellow. Check out the comments in the
#       code itself.
#       link: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
#______________________________________________________________________________________________________________________________________________
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

#______________________________________________________________________________________________________________________________________________
#explain:
#   -> Base on a filename, a path and an element of the dict_label (output of csv2labelDict), it creates a dictinoray with the JSON
#       template provided in Json_sample.json.
#arg:
#   name: file name
#   element: an element of dict_lable (output of csv2labelDict)
#   pth: save path of the image
#
#return:
#   (dict) -> Coresponding to Json_sample.json file
#______________________________________________________________________________________________________________________________________________
def _create_jsondict( name, elements, pth ):
    # read element
    img = cv2.imread( path.join(pth , name) )

    color_mode = _find_colormode(img)
    img_shape = _get_img_shape(img)
    label_type = "MASK"
    labels = []
    for idx , cls in enumerate(elements[0]):
        labels_list = elements[1][idx].tolist()
        dic_tmp = {'class': cls , 'mask': labels_list}
        labels.append(dic_tmp)

    return {
        'name': name,
        'path': pth,
        'color_mode': color_mode,
        'size': list(img_shape),
        'included_object': 'YES',
        'label_type': 'MASK',
        'labels' : labels
    }



#______________________________________________________________________________________________________________________________________________
#explain:
#   -> Gets an output path and a dict var (logicaly the output of _create_jsondict) and saves the JSON of that dict in the provided 
#       path. If singleLine == True, it dosen't format the JSON and put's everything in one line.
#arg:
#   jsn: an input dictionary which decribes the value to be written
#   save_path: location to store the json.
#   singleLine: bool, True if you want to save everything in a single line.
#return:
#   
#______________________________________________________________________________________________________________________________________________
def _save_json( jsn , save_path , singleLine = False):
    json_name = path.splitext(jsn['name'])[0] + ".json"
    with open(path.join(save_path , json_name) , 'w') as jsfile:
        if singleLine:
            json.dump(jsn , jsfile)
        else:   
            json.dump(jsn , jsfile , indent= 4)


#______________________________________________________________________________________________________________________________________________
#explain:
#   -> Creates apropritae JSON files, using the train.csv file
#arg:
#   images_path: path where the images are stored
#   csv_path: location of train.csv file
#   save_path: where to store jsons
#   singleLine: bool, True if you want to save jsons with single line.
#return:
#   
#______________________________________________________________________________________________________________________________________________
def convert_csv_to_json(images_path , csv_path , save_path, singleLine = False):

    csv_file = csv_reader(csv_path)
    dict_lbl = csv2labelDict(csv_file)

    print("Conversion Started. Please Wait...")
    counter = 1
    total = len(dict_lbl.items())
    for key, val in dict_lbl.items():
        printProgressBar (counter, total, prefix = f'{"Converting CSV to JSON": <25}', suffix = 'Completed', decimals = 1, length = 100, fill = '█', printEnd = "\r")
        json_dict = _create_jsondict(key , val , images_path)
        _save_json(json_dict , save_path, singleLine)   
        counter += 1


#______________________________________________________________________________________________________________________________________________
#explain:
#   -> Using the json files saved by the convert_csv_to_jsons, it compares all the images with the saved json files and creates new,
#       jsons with empyt label and "included_object: NO" for the unlabeled images.
#arg:
#   image_path: path where the images are stored
#   json_path: location where the json files are stored
#   singleLine: bool, True if you want to save jsons with single line.
#
#return:
#   
#______________________________________________________________________________________________________________________________________________
def create_json_for_unlabeled_image(image_path , json_path , singleLine = False):

    json_path_list = os.listdir(json_path)
    image_path_list = os.listdir(image_path)

    def reduce_path(filename , pth):
        if path.isfile(path.join(pth,filename)):
            fn_extended = path.splitext(filename)

            return fn_extended[:-1][0] , fn_extended[-1] 
    
    json_path_list_reduced = list(
        map(lambda inp: reduce_path(inp , json_path)[0] , json_path_list)
    )

    image_path_list_reduced = list(
        map(lambda inp: reduce_path(inp , image_path) , image_path_list)
    )

    # print(json_path_list_reduced)
    total = len(image_path_list_reduced)
    iteration = 1

    print("\n\nCreating label for unlabled images. Please Wait...")
    for i_fname , i_fname_extension in image_path_list_reduced:
        printProgressBar (iteration, total, prefix = f'{"Creating JSONS": <25}', suffix = 'Completed', decimals = 1, length = 100, fill = '█', printEnd = "\r")
        if  not i_fname in json_path_list_reduced:

            img = cv2.imread( path.join(image_path , i_fname + i_fname_extension) )
            color_mode = _find_colormode(img)
            img_shape = _get_img_shape(img)

            temp_dic = {
                'name': i_fname + i_fname_extension,
                'path': image_path,
                'color_mode': color_mode,
                'size': list(img_shape),
                'included_object': 'NO',
                'label_type': '',
                'labels' : ''
            }

            _save_json(temp_dic , json_path , singleLine = singleLine)
        iteration += 1


#______________________________________________________________________________________________________________________________________________
#explain:
#   -> Returns the size of the image
#
#arg:
#   img: ( np.ndarray() ) input image file
#
#return:
#   tuple(height , width)
#______________________________________________________________________________________________________________________________________________
def _get_img_shape( img ):
    return img.shape[:2]



#______________________________________________________________________________________________________________________________________________
#explain:
#   -> Returns the color mode, "COLOR" for colored images and "GRAY" for grayscale images by comparing color chanels to each other.
#
#arg:
#   image: ( np.ndarray() ) input image file
#   threshold:  the threshold error for comparing image chanels.
#
#return:
#   "GRAY" for grayscale images and "COLOR" for colored images
#______________________________________________________________________________________________________________________________________________
def _find_colormode( image , threshold = 20 ):
    b_g = np.abs( image[:,:,0] - image[:,:,1] )
    g_r = np.abs( image[:,:,1] - image[:,:,2] )
    b_r = np.abs( image[:,:,0] - image[:,:,2] )

    if np.sum(b_g) < threshold and np.sum(g_r) < threshold and np.sum(b_r) < threshold:
        return "GRAY"
    
    else:
        return "COLOR"


#______________________________________________________________________________________________________________________________________________
#explain:
#   -> Creates all json files for the dataset
#arg:
#   images_path: path where the images are stored
#   csv_path: location of train.csv file
#   json_save_path: where to store jsons
#   singleLine: bool, True if you want to save jsons with single line.
#return:
#   
#______________________________________________________________________________________________________________________________________________
def create_all_json(image_path , csv_path , json_save_path , singleLine = False):

    os.system('clear')
    convert_csv_to_json(image_path , csv_path , json_save_path , singleLine = singleLine)
    create_json_for_unlabeled_image(image_path , json_save_path , singleLine = singleLine )
    print("\n","\033[92m" + ".::Conversion Finished Successfully!::.\n" + '\033[0m')

def main():
    # _pretify_lists(r'.\severstal-steel-defect-detection\annotations\000a4bcdd.json')

    images_path = r"./severstal-steel-defect-detection/train_images"
    csv_path = r"./severstal-steel-defect-detection/train.csv"
    json_save_path = r"./severstal-steel-defect-detection/annotations"
    create_all_json(images_path , csv_path , json_save_path)

if __name__ == "__main__":
    main()