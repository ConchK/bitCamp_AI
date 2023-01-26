from tensorflow.keras.datasets import cifar100
import numpy as np
#1.데이터

(x_train, y_train), (x_test, y_test)= cifar100.load_data()

print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     #(10000, 32, 32, 3) (10000, 1)   >이미 4차원이기 때문에 reshape필요 없음.


print(np.unique(y_train, return_counts=True))
#(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    #    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    #    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #    500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))

from sklearn.preprocessing import MinMaxScaler

class MinMaxScaler3D(MinMaxScaler):

    def fit_transform(self, x, y=None):
        x = np.reshape(x, newshape=(x.shape[0]*x.shape[1], x.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=x.shape)

print(x_train.shape, x_test.shape)

#2.모델

from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Conv2D, Dense, Flatten, MaxPooling2D

model= Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(32,32,3), padding= 'same', activation= 'relu'))    #(31,31,128)
model.add(Conv2D(filters=64, kernel_size=(2,2), padding= 'same'))    #(30,30,64)
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2)))    #(29,29,64)
model.add(MaxPooling2D())
model.add(Flatten())    #53824
model.add(Dense(200, activation= 'relu'))   
model.add(Dense(100, activation= 'softmax'))
            #마지막 output은 y_train의 갯수
#3.컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k34_3_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  

# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name) 

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_split=0.2, callbacks=[earlystopping])

#4 평가
results= model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

'''
loss:  4.605268955230713
acc:  0.009999999776482582
'''