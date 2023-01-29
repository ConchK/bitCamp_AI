import numpy as np
from tensorflow.keras.datasets import mnist

#1.데이터

(x_train, y_train), (x_test, y_test)= mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

#이미지는 3차원 > 4차원으로 바꿔준다.
x_train= x_train.reshape(60000,28,28,1) #60000 행, (28,28,1)열
x_test= x_test.reshape(10000,28,28,1)

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))


#2 모델

from tensorflow. keras. models import Sequential, Model
from tensorflow. keras. layers import Conv2D, Dense, Flatten, MaxPooling2D, Input 

# model= Sequential()
# model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28,28,1),
#                  padding= 'same',
#                  strides=2,  # 연산으로 2칸씩 넘어감. Maxpooling2D 랑은 다름. 1 = 1칸씩.
#                  activation= 'relu'))    #(28,28,128)
# model.add(MaxPooling2D())   # max_pooling2d (MaxPooling2D  (None, 14, 14, 128)      0) 
# model.add(Conv2D(filters=64, kernel_size=(2,2),
#                  padding = 'same'))  
# model.add(Conv2D(filters=64, kernel_size=(2,2)))    #(13,13,64)
# model.add(Flatten())    #10816
# model.add(Dense(32, activation= 'relu'))        #input_shape= (40000,)
# model.add(Dense(10, activation= 'softmax'))

# model.summary()

#2. 모델구성(함수형)  
input1= Input(shape=(28,28,1))
conv1= Conv2D(filters=128, kernel_size=(3,3), input_shape=(28,28,1),
        padding= 'same', strides=2, activation= 'relu')(input1)
maxpool1= MaxPooling2D()(conv1)
conv2= Conv2D(filters=64, kernel_size=(2,2), padding = 'same')(maxpool1)
conv3= Conv2D(filters=64, kernel_size=(2,2))(conv2)
flatten= Flatten()(conv3)
dense1= Dense(32, activation= 'relu')(flatten)
output1= Dense(10, activation= 'softmax')(dense1)
model= Model(inputs=input1, outputs=output1)   #정의가 마지막으로 
model.summary()
#Total params: 3,696

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

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_split=0.2, callbacks=[earlystopping])

#4 평가
results= model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

'''
loss:  0.09927216172218323
acc:  0.9733999967575073
'''