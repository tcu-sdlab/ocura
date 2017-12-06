import os
import numpy as np
import cv2 as cv
import gc

data_dir_path = "dataset_images/new/"
tmp = os.listdir(data_dir_path)
tmp=sorted([x for x in tmp if os.path.isdir(data_dir_path+x)])
dir_list = tmp

X_target = np.empty((0, 3, 32, 32), dtype=np.uint8)
c_class = []

for dir_name in dir_list:
    print(dir_name)
    file_list = os.listdir(data_dir_path+dir_name)

    png_count = 0
    for f in file_list:
        if f.endswith('.png'):
            png_count+=1

    print(png_count)

    X_tmp = np.empty((png_count, 3, 32, 32), dtype=np.uint8)

    png_count = 0
    for file_name in file_list:
        if file_name.endswith('.png'):
            image_path=str(data_dir_path)+str(dir_name)+'/'+str(file_name)
            image = cv.imread(image_path)
            image = cv.resize(image, (32, 32))  #[32][32][3]
            image = image.transpose(2,0,1)  #[3][32][32]
            image = image/255.
            tmp = np.array([image])
            X_tmp[png_count] = tmp
            png_count+=1

            c_class.append(dir_name)
            # print(file_name)

    X_target = np.vstack((X_target, X_tmp))


if not os.path.exists('npy_data'):
    os.mkdir('npy_data')

c_arr=np.array(c_class)
np.save('npy_data/character_label_new.npy',c_arr)

try:
    np.save('npy_data/character_data_new.npy',X_target)
except MemoryError as e:
    print(e)


