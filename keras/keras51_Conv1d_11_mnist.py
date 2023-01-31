import numpy as np
from tensorflow.keras.datasets import mnist

#1.데이터

(x_train, y_train), (x_test, y_test)= mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

#3차원 이상부터는 y의 열의 개수를 알기 위해 사용.
print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))


#2 모델

from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Conv1D, Dense, Flatten, Dropout

model= Sequential()
model.add(Conv1D(filters=40, kernel_size= 4, input_shape=(28,28), strides= 2, activation= 'relu'))    
model.add(Conv1D(filters=30, kernel_size= 4, activation = 'relu')) 
model.add(Conv1D(filters=20, kernel_size= 4, activation = 'relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(60, activation= 'relu'))  
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k34_1_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  

mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name) 

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_split=0.2, callbacks=[earlystopping, mcp])

#4 평가
results= model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

'''
Conv2D
loss:  2.3012239933013916
acc:  0.11349999904632568
Conv1D

'''

#early, mcp, val 적용.