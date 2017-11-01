# coding: utf-8

import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
import sklearn.cross_validation
import cv2 as cv
from PIL import Image
import sys
from scipy import ndimage
import tensorflow as tf
import os
from operator import itemgetter
import math

def get_best_shift(img):
  #ここでnanが出ることがある
  cy,cx = ndimage.measurements.center_of_mass(img)
  print('{},{}'.format(cx,cy))

  rows,cols = img.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return shiftx,shifty


def shift(img,sx,sy):
  rows,cols = img.shape
  M = np.float32([[1,0,sx],[0,1,sy]])
  shifted = cv.warpAffine(img,M,(cols,rows))

  return shifted


def processing(img):
  img = cv.resize(img, (32, 32))
  img = img.transpose(2,0,1)
  img = img/255.
  img = img.reshape(1,3,32,32)

  return img


def resize_in_28x(img):
  rows,cols = img.shape
  compl_dif = abs(rows-cols)
  half_Sm = compl_dif/2
  half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
  if rows > cols:
    img = np.lib.pad(img,((0,0),(half_Sm,half_Big)),'constant')
  else:
    img = np.lib.pad(img,((half_Sm,half_Big),(0,0)),'constant')

  img = cv.resize(img, (20, 20))
  img = np.lib.pad(img,((4,4),(4,4)),'constant')

  return img


def recognition(model, image):
  #学習データと比較して損失が最も少ないものを認識結果として出す
  max_score = 0
  answer_i = 0

  for i in range(63):
    sample_target=np.array([i])
    score = model.evaluate(image, sample_target, verbose=0)

    if not char_arr[i]=='':
      percent=(16.12-score[0])/16.12*100  #確率を導出(損失の最大値がおおよそ16.12)
      # print('{0} : {1:.2f}%'.format(char_arr[i], percent))

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
  cv.putText(color_complete,str(pred),(top_left[1],bottom_right[0]+50),font,fontScale=1.4,color=(0,255,0),thickness=4)
  cv.putText(color_complete,format(max_score,".2f")+"%",(top_left[1],bottom_right[0]+80),font,fontScale=0.8,color=(0,255,0),thickness=2)

  return pred


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

  return arr


def create_textfile(arr, num_arr):
  f = open('recog_images/pro_img/{0}/{0}.txt'.format(image_name), 'w')

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
  return


np.random.seed(20171019)
sess = tf.Session()

#文字認識の学習データ
X_test=np.load('character_data.npy')
Y_target=np.load('character_label.npy')
#学習したモデル
model = model_from_json(open('character_recog_model.json').read())
model.load_weights('character_recog_model.h5')
#認識の準備
init_learning_rate = 1e-2
opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["acc"])

#画像の読み込みと加工
image_name = sys.argv[1]    #画像の名前
image_format = sys.argv[2]  #画像のフォーマット
image_url = "/".join(['test_images', image_name+'.'+image_format] )
image = cv.imread(image_url,0)
color_complete = image
_, image = cv.threshold(255-image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
digit_image = -np.ones(image.shape)

#認識結果を文字で表示するための配列
char_arr = ['0','1','2','3','4','5','6','7','8','9','','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','','','','+','-','*','/','%','=','space','#','$','^','?','back_slash',':',';',',','.','@','(',')','[',']','<','>']

#認識結果を格納するための配列
result_arr = []
line_num_arr = []

#読み込んだ画像の高さと横幅
height = image.shape[0]
width = image.shape[1]

#画像を出力するためのフォルダを作成
if not os.path.exists('recog_images/pro_img'):
  os.mkdir('recog_images/pro_img')
if not os.path.exists('recog_images/pro_img/{}'.format(image_name)):
  os.mkdir('recog_images/pro_img/{}'.format(image_name))

#画像内から文字が記述されている範囲を特定するためのループ
for cropped_width in range(100/2, 300/2, 20/2):
    for cropped_height in range(100/2, 300/2, 20/2):
        for shift_x in range(0, width-cropped_width, cropped_width/4):
            for shift_y in range(0, height-cropped_height, cropped_height/4):

                #特定のサイズに切り出す
                gray = image[shift_y:shift_y+cropped_height,shift_x:shift_x + cropped_width]

                #色がついている部分がなければ次の範囲へ
                if np.count_nonzero(gray) <= 40:
                	# print('continue1')
                	continue

                #桁を切り取って白い枠線が見つからない箇所があれば次の範囲へ
                if (np.sum(gray[0]) != 0) or (np.sum(gray[:,0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:,-1]) != 0):
                    continue

                #切り取った範囲の左上と右下の座標
                top_left = np.array([shift_y, shift_x])
                bottom_right = np.array([shift_y+cropped_height, shift_x + cropped_width])

                #文字の位置を特定
                while np.sum(gray[0]) == 0:
                    top_left[0] += 1
                    gray = gray[1:]

                while np.sum(gray[:,0]) == 0:
                    top_left[1] += 1
                    gray = np.delete(gray,0,1)

                while np.sum(gray[-1]) == 0:
                    bottom_right[0] -= 1
                    gray = gray[:-1]

                while np.sum(gray[:,-1]) == 0:
                    bottom_right[1] -= 1
                    gray = np.delete(gray,-1,1)

                #既に文字が認識されている範囲であれば次の範囲へ
                actual_w_h = bottom_right-top_left
                if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]+1) >
                            0.05*actual_w_h[0]*actual_w_h[1]):
                    continue

                #画像を28x28のサイズに成形する
                gray = resize_in_28x(gray)

                #重心を使用してシフト
                shiftx,shifty = get_best_shift(gray)
                if math.isnan(shiftx):
                  cv.imwrite("recog_images/pro_img/{}/{}_{}_{}_{}_nan_image.png".format(image_name,cropped_width,cropped_height,shift_x,shift_y), gray)
                gray = shift(gray,shiftx,shifty)

                #画像を書き出してからもう一度読み込み
                #(シフト後の画像は2次元配列に入っているが3次元じゃないと文字認識ができない)
                shifted_image_url = "recog_images/pro_img/{}/{}_{}_{}_{}_shifted_image.png".format(image_name,cropped_width,cropped_height,shift_x,shift_y)
                cv.imwrite(shifted_image_url, gray)
                image_tmp = cv.imread(shifted_image_url)
                image_tmp = processing(image_tmp)

                flatten = gray.flatten() / 255.0

                #認識結果
                pred = recognition(model, image_tmp)
                # print('our prediction is ... {}'.format(pred))

                #認識結果を配列に格納
                result_arr.append([pred,shift_x,shift_y,bottom_right[1],bottom_right[0]])

#画像を出力
cv.imwrite("recog_images/pro_img/{0}/{0}_digitized_image.png".format(image_name), color_complete)

#結果を格納した配列を文字の座標で列ごとにソート
sorted_arr = sort_by_line(result_arr)

#テキストファイル生成
create_textfile(sorted_arr, line_num_arr)

for s in sorted_arr:
  print('{} {},{} line:{}'.format(s[0],s[1],s[2],s[5]+1))
