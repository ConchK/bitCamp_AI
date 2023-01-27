import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])


#이미지는 3차원 > 4차원으로 바꿔준다.
x_train= x_train.reshape(60000,28,28,1) #60000 행, (28,28,1)열
x_test= x_test.reshape(10000,28,28,1)


print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))


#2 모델

from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Conv2D, Dense, Flatten

model= Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), input_shape=(28,28,1), activation= 'relu'))    #(27,27,128)
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

# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name) 

model.fit(x_train, y_train, epochs=100, verbose=2, batch_size=32, validation_split=0.2, callbacks=[earlystopping])

#4 평가
results= model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])


import matplotlib.pyplot as plt
plt.imshow(x_train[0], 'gray')    #밝은 부분이 데이터가 높음
plt.show()

'''
loss:  1.6845507621765137
acc:  0.25699999928474426
'''
