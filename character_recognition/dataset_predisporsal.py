import os
import numpy as np
import cv2 as cv
import gc

def get_dir_list():
  data_dir_path = "dataset_images/lower/"
  dir_list = os.listdir(data_dir_path)
  dir_list = sorted([x for x in dir_list if os.path.isdir(data_dir_path+x)])

  return (data_dir_path, dir_list)


def image_processing(image_path):
  image = cv.imread(image_path)
  image = cv.resize(image, (32, 32))  #[32][32][3]
  image = image.transpose(2,0,1)  #[3][32][32]
  image = image/255.

  return image


def make_target_and_class(data_dir_path, dir_list):
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
        image_path = os.path.join(str(data_dir_path)+str(dir_name),str(file_name))
        image = image_processing(image_path)
        X_tmp[png_count] = np.array([image])
        png_count+=1

        c_class.append(dir_name)
        # print(file_name)

    X_target = np.vstack((X_target, X_tmp))

  return (c_class, X_target)


def main():
  data_dir_path, dir_list = get_dir_list()

  c_class, X_target = make_target_and_class(data_dir_path, dir_list)

  if not os.path.exists('npy_data'):
    os.mkdir('npy_data')

  c_arr=np.array(c_class)
  np.save('npy_data/character_label_lower.npy',c_arr)

  try:
    np.save('npy_data/character_data_lower.npy',X_target)
  except MemoryError as e:
    print(e)


if __name__ == '__main__':
  main()
