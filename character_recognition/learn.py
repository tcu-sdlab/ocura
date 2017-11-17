import numpy as np
np.random.seed(20160715)
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.optimizers import SGD
import sklearn.cross_validation

X_test=np.load('character_data.npy')
Y_target=np.load('character_label.npy')

a_train, a_test, b_train, b_test = sklearn.cross_validation.train_test_split(X_test,Y_target)

model = Sequential()

model.add(Conv2D(96, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
model.add(Activation('relu'))

model.add(Conv2D(128, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(203))
model.add(Activation('softmax'))

init_learning_rate = 1e-2
opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["acc"])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
lrs = LearningRateScheduler(0.01)

hist = model.fit(a_train,b_train, 
                batch_size=128, 
                epochs=23, 
                validation_split=0.1, 
                verbose=1)

model_json_str = model.to_json()
open('character_recog_model.json', 'w').write(model_json_str)
model.save_weights('character_recog_model.h5')

score=model.evaluate(a_test, b_test, verbose=0)
print(score[1])