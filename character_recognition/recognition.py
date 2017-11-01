import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
import sklearn.cross_validation
import cv2 as cv
from PIL import Image
import sys

np.random.seed(20171019)

X_test=np.load('character_data.npy')
Y_target=np.load('character_label.npy')

model = model_from_json(open('character_recog_model.json').read())
model.load_weights('character_recog_model.h5')
init_learning_rate = 1e-2
opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["acc"])

image = cv.imread("/".join(['test_images', sys.argv[1]] ))
image = cv.bitwise_not(image)
image = cv.resize(image, (32, 32))
image = image.transpose(2,0,1)
image = image/255.
image=image.reshape(1,3,32,32)

char_Arr = ['0','1','2','3','4','5','6','7','8','9','','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','','','','+','-','*','/','%','=','space','#','$','^','?','back_slash',':',';',',','.','@','(',')','[',']','<','>']

max_score = 0
answer_i = 0

for i in range(63):
    sample_target=np.array([i])
    score = model.evaluate(image, sample_target, verbose=0)

    if not char_Arr[i]=='':
    	percent=(16.12-score[0])/16.12*100
    	print('{0} : {1:.2f}%'.format(char_Arr[i], percent))

    if percent>max_score:
    	max_score = percent
    	answer_i = i

    # if score[1]==1.0:
    #     break


print(char_Arr[answer_i])