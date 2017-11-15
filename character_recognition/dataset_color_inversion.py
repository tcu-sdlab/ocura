import os
import numpy as np
import cv2 as cv
from PIL import Image
from scipy import ndimage

if not os.path.exists('dataset_images/new'):
    os.mkdir('dataset_images/new')

data_dir_path = "dataset_images/new_tmp/"
tmp = os.listdir(data_dir_path)
tmp=sorted([x for x in tmp if os.path.isdir(data_dir_path+x)])
dir_list = tmp

print(dir_list)

for dir_name in dir_list:
    file_list = os.listdir(data_dir_path+dir_name)
    if not os.path.exists('dataset_images/new/{}/'.format(dir_name)):
        os.mkdir('dataset_images/new/{}'.format(dir_name))

    for (i, file_name) in enumerate(file_list):
        if file_name.endswith('.png'):
            image_path=str(data_dir_path)+str(dir_name)+'/'+str(file_name)
            image = cv.imread(image_path, 0)
            image = cv.resize(image, (32, 32))
            # image = image.transpose(2,0,1)
            _, image = cv.threshold(255-image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            cv.imwrite('dataset_images/new/{0}/{0}_{1}.png'.format(dir_name, i), image)