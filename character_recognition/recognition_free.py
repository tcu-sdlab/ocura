# coding: utf-8

import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
import sklearn.cross_validation
import cv2 as cv
from PIL import Image
from scipy import ndimage
import tensorflow as tf
import os
from operator import itemgetter
import math


def add_padding(image, tl_br):
  top_left, bottom_right = tl_br

  if(image.shape[0]<50 or image.shape[1]<50):
    for (i, (br, tl)) in enumerate(zip(bottom_right, top_left)):
      if (image.shape[i]<50):
        pad_size = ((50-image.shape[i])/2)
        bottom_right[i] += pad_size
        top_left[i] -= pad_size
        image = np.lib.pad(image,(pad_size,pad_size),'constant', constant_values=(0,0))

  return image


def create_directory(path):
  if not os.path.exists(path):
    os.mkdir(path)


def create_textfile(arr, num_arr, image_name):
  file_url = os.path.join('recog_images', 'pro_images', image_name, '{}.txt'.format(image_name))
  f = open(file_url, 'w')

  sum_value = 0
  for i in range(arr[len(arr)-1][5]+1):
    out_arr = []
    slice_arr = arr[sum_value:sum_value+num_arr.count(i)]
    for s in slice_arr:
      out_arr.append(s[0])

    sum_value+=num_arr.count(i)
    print(sum_value)
    print(out_arr)
    out = ''.join(out_arr)
    f.write('{}\n'.format(out))

  f.close()


def delete_margin(image, tl_br):
  top_left, bottom_right = tl_br

  while np.sum(image[0]) == 0:
    top_left[0] += 1
    image = image[1:]

  while np.sum(image[:,0]) == 0:
    top_left[1] += 1
    image = np.delete(image,0,1)

  while np.sum(image[-1]) == 0:
    bottom_right[0] -= 1
    image = image[:-1]

  while np.sum(image[:,-1]) == 0:
    bottom_right[1] -= 1
    image = np.delete(image,-1,1)

  return (image, top_left, bottom_right)


def export_and_import_image(image_name, image, cwh_and_sxy):
  cropped_width, cropped_height, shift_x, shift_y = cwh_and_sxy

  shifted_image_url = os.path.join('recog_images', 'pro_images', image_name, 'shift', '{}_{}_{}_{}_shifted_image.png'.format(cropped_width,cropped_height,shift_x,shift_y))
  cv.imwrite(shifted_image_url, image)
  image_tmp = cv.imread(shifted_image_url)
  image_tmp = processing(image_tmp)

  return image_tmp


def export_crop_image(image_name, image1, image2, cwh_and_sxy):
  cropped_width, cropped_height, shift_x, shift_y = cwh_and_sxy
  crop_directory_url = os.path.join('recog_images', 'pro_images', image_name, 'crop')
  cv.imwrite(os.path.join(crop_directory_url,'{}_{}_{}_{}_first_image.png'.format(cropped_width,cropped_height,shift_x,shift_y)), image1)
  cv.imwrite(os.path.join(crop_directory_url,'{}_{}_{}_{}_second_image.png'.format(cropped_width,cropped_height,shift_x,shift_y)), image2)


def export_result(image_name, image, sorted_arr, line_num_arr):
  #画像を出力
  digitized_image_url = os.path.join('recog_images', 'pro_images', image_name, '{}_digitized_image.png'.format(image_name))
  cv.imwrite(digitized_image_url, image)
  #テキストファイル生成
  create_textfile(sorted_arr, line_num_arr, image_name)


def get_best_shift(image):
  #ここでnanが出ることがある
  cy,cx = ndimage.measurements.center_of_mass(image)
  #print('{},{}'.format(cx,cy))

  rows,cols = image.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return (shiftx,shifty)


def is_character_in_range(image):
  #画像内に白い部分がない場合(何も書かれていない範囲)
  if np.count_nonzero(image) <= 20:
    return True

  #文字らしきものが一番外のふちにあった場合(文字が切れている可能性)
  if (np.sum(image[0]) != 0) or (np.sum(image[:,0]) != 0) or (np.sum(image[-1]) != 0) or (np.sum(image[:,-1]) != 0):
    return True

  return False


def is_recognized_range(digit_image, tl_br):
  top_left, bottom_right = tl_br
  actual_w_h = bottom_right - top_left

  if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]+1) > 0.03*actual_w_h[0]*actual_w_h[1]):
    return True

  return False


def preparing_directories(image_name):
  create_directory('recog_images')
  pro_images_directory_url = os.path.join('recog_images','pro_images')
  create_directory(pro_images_directory_url)
  image_directory_url = os.path.join(pro_images_directory_url, image_name)
  create_directory(image_directory_url)
  crop_directory_url = os.path.join('recog_images', 'pro_images', image_name, 'crop')
  create_directory(crop_directory_url)
  shift_directory_url = os.path.join('recog_images', 'pro_images', image_name, 'shift')
  create_directory(shift_directory_url)


def preparing_image():
  import sys

  image_name = sys.argv[1]    #画像の名前
  image_format = sys.argv[2]  #画像のフォーマット
  image_url = os.path.join('recog_images', 'test_images', image_name+'.'+image_format)
  print(image_url)
  image = cv.imread(image_url,0)
  image = cv.resize(image, (2000, 2830))
  color_complete = image
  _, image = cv.threshold(255-image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
  digit_image = -np.ones(image.shape)

  return(image_name, image_format, image, color_complete, digit_image)


def preparing_model():
  model_url = os.path.join('model', 'new', 'character_recog_model_nist.json')
  weight_model_url = os.path.join('model', 'new', 'character_recog_model_nist.h5')
  #学習したモデルの読み込みと準備
  model = model_from_json(open(model_url).read())
  model.load_weights(weight_model_url)
  init_learning_rate = 1e-2
  opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9, nesterov=False)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["acc"])

  return model


def processing(image):
  image = cv.resize(image, (32, 32))
  image = image.transpose(2,0,1)
  image = image/255.
  image = image.reshape(1,3,32,32)

  return image


def recognition(model, image):
  #学習データと比較して損失が最も少ないものを認識結果として出す
  max_score = 0
  answer_i = 0

  for i in range(93):
    sample_target=np.array([i])
    score = model.evaluate(image, sample_target, verbose=0)

    if not char_arr[i]=='':
      percent=(16.12-score[0])/16.12*100  #確率を導出(損失の最大値がおおよそ16.12)
      print('{0}({2}) : {1:.2f}%'.format(char_arr[i], percent, i))

    if percent>max_score:
      max_score = percent
      answer_i = i

  #認識結果
  pred = char_arr[answer_i]

  #文字が認識できた範囲として配列の要素にすべて1を格納する(0が認識されていないという状態)
  digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]] = np.ones_like(digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]])

  #文字がある範囲に四角を描画
  cv.rectangle(color_complete,tuple(top_left[::-1]),tuple(bottom_right[::-1]),color=(0,255,0),thickness=5)

  #認識結果と確率を四角の下に描画
  font = cv.FONT_HERSHEY_SIMPLEX
  cv.putText(color_complete,str('{}({})'.format(pred, answer_i)),(top_left[1],bottom_right[0]+50),font,fontScale=1.4,color=(0,255,0),thickness=4)
  cv.putText(color_complete,format(max_score,".2f")+"%",(top_left[1],bottom_right[0]+80),font,fontScale=0.8,color=(0,255,0),thickness=2)

  return pred


def recognition_free(image, digit_image, image_name):
  #認識用のモデル
  model = preparing_model()
  #読み込んだ画像の高さと横幅
  height = image.shape[0]
  width = image.shape[1]

  #画像内から文字が記述されている範囲を特定するためのループ
  for cropped_width in range(60, 120, 5):
    for cropped_height in range(50, 120, 5):  #ここを固定にしたい
      for shift_x in range(280, (width-280)-cropped_width, cropped_width/5):
        for shift_y in range(230, (height-230)-cropped_height, cropped_height/5):

          cwh_and_sxy = (cropped_width, cropped_height, shift_x, shift_y)

          #特定のサイズに切り出す
          gray = image[shift_y:shift_y+cropped_height, shift_x:shift_x+cropped_width]
          #切り取った範囲の左上と右下の座標
          top_left = np.array([shift_y, shift_x])
          bottom_right = np.array([shift_y+cropped_height, shift_x + cropped_width])
          tl_br = (top_left, bottom_right)

          gray_origin = gray

          #色がついている部分がないか、文字が画像内に収まっていなければ次の範囲へ
          if is_character_in_range(gray):
            continue

          #余白を削る
          gray, top_left, bottom_right = delete_margin(gray, tl_br)

          #画像が小さすぎる時に50pxまで補填
          gray = add_padding(gray, tl_br)

          #既に文字が認識されている範囲であれば次の範囲へ
          if (is_recognized_range(digit_image, tl_br)):
            continue

          #切り出した画像を出力
          export_crop_image(image_name, gray_origin, gray, cwh_and_sxy)

          #画像を28x28のサイズに成形する
          gray = resize_in_28x(gray)

          #重心を使用してシフト
          shiftx,shifty = get_best_shift(gray)
          gray = shift(gray,shiftx, shifty)

          #画像を書き出してからもう一度読み込み
          #(シフト後の画像は2次元配列に入っているが3次元じゃないと文字認識ができない)
          gray = export_and_import_image(image_name, gray, cwh_and_sxy)

          #認識結果
          pred = recognition(model, gray)
          # print('our prediction is ... {}'.format(pred))

          #認識結果を配列に格納
          result_arr.append([pred,shift_x,shift_y,bottom_right[1],bottom_right[0]])

  return result_arr


def resize_in_28x(image):
  rows,cols = image.shape
  compl_dif = abs(rows-cols)
  half_Sm = compl_dif/2
  half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
  if rows > cols:
    image = np.lib.pad(image,((0,0),(half_Sm,half_Big)),'constant')
  else:
    image = np.lib.pad(image,((half_Sm,half_Big),(0,0)),'constant')

  image = cv.resize(image, (20, 20))
  image = np.lib.pad(image,((4,4),(4,4)),'constant')

  return image


def shift(image,sx,sy):
  rows,cols = image.shape
  M = np.float32([[1,0,sx],[0,1,sy]])
  shifted = cv.warpAffine(image,M,(cols,rows))

  return shifted


def sort_by_line(arr):
  arr = sorted(arr, key=lambda x:(x[2],x[1]))
  height_tmp = arr[0][4] - arr[0][2]
  y_tmp = arr[0][2]

  line_num = 0
  for (i, r) in enumerate(arr):
    if not (r[2]<=(y_tmp+height_tmp)):
      height_tmp = r[4] - r[2]
      y_tmp = r[2]
      line_num += 1

    # result_arr[i][2] = y_tmp
    arr[i].append(line_num)
    line_num_arr.append(line_num)

  arr = sorted(result_arr, key=lambda x:(x[5],x[1]))

  return (arr, line_num_arr)


def main():
  #画像の読み込みと加工
  image_name, image_format, image, color_complete, digit_image = preparing_image()

  #画像を出力するためのフォルダを作成
  preparing_directories(image_name)

  #認識結果を文字で表示するための配列
  char_arr = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','','','','','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','','','','','+','-','*','/','%','=','space','#','$','^','?','apostrophe',':',';',',','.','@','(',')','[',']','<','>']
  #認識結果を格納するための配列
  result_arr = []
  #文字ごとの列を格納するための配列
  line_num_arr = []

  #文字の認識
  result_arr = recognition_free(image, digit_image, image_name)

  #結果を格納した配列を文字の座標で列ごとにソート
  sorted_arr, line_num_arr = sort_by_line(result_arr)

  #結果の出力
  export_result(image_name, color_complete, sorted_arr, line_num_arr)

  for s in sorted_arr:
    print('{} {},{} line:{}'.format(s[0],s[1],s[2],s[5]+1))


if __name__ == '__main__':
  main()