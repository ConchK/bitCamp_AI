from tensorflow.keras.datasets import cifar10
import numpy as np
#1.데이터

(x_train, y_train), (x_test, y_test)= cifar10.load_data()

print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     #(10000, 32, 32, 3) (10000, 1)   >이미 4차원이기 때문에 reshape필요 없음.


from sklearn.preprocessing import MinMaxScaler

class MinMaxScaler3D(MinMaxScaler):

    def fit_transform(self, x, y=None):
        x = np.reshape(x, newshape=(x.shape[0]*x.shape[1], x.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=x.shape)

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

#2.모델

from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Conv2D, Dense, Flatten, MaxPooling2D

model= Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32,32,3), padding = 'same', activation= 'relu'))    #(31,31,128)
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2), padding= 'same', strides=2))    #(30,30,64)
model.add(Conv2D(filters=64, kernel_size=(2,2), padding= 'same'))    #(29,29,64)
model.add(Flatten())    #53824
model.add(Dense(32, activation= 'relu'))    
model.add(Dense(10, activation= 'softmax'))

#3.컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k34_2_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  

# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name) 

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_split=0.2, callbacks=[earlystopping])

#4 평가
results= model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

'''
loss:  2.3025941848754883
acc:  0.10000000149011612
'''