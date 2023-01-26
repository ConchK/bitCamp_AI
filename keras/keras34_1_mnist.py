import numpy as np
from tensorflow.keras.datasets import mnist

#1.데이터

(x_train, y_train), (x_test, y_test)= mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

#이미지는 3차원 > 4차원으로 바꿔준다.
x_train= x_train.reshape(60000,28,28,1) #60000 행, (28,28,1)열
x_test= x_test.reshape(10000,28,28,1)

# numpy로 차원 추가
# x_train= np.expand_dims(x_train, -1)    # x_train 위치에 변경하려는 배열을 넣어준다. -1은 가장 뒷부분의 차원을 추가

# numpy로 차원 제거
# x_train = np.squeeze(x_train, [0])  #  np.squeeze 내부에는 축소하고 싶은 인덱스를 넣어주면 된다. 1차원인 축을 모두 제거
# print(x_train.shape)
# '''
# (28, 28)
# '''

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))


#2 모델

from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Conv2D, Dense, Flatten

model= Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28,28,1), activation= 'relu'))    #(27,27,128)
model.add(Conv2D(filters=64, kernel_size=(2,2)))    #(26,26,64)
model.add(Conv2D(filters=64, kernel_size=(2,2)))    #(25,25,64)
model.add(Flatten())    #40000
model.add(Dense(32, activation= 'relu'))        #input_shape= (40000,)
                     #(60000,40000) 이 인풋.  (batch_size, input_dim)
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
loss:  2.3012239933013916
acc:  0.11349999904632568
'''

#early, mcp, val 적용.