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
from tensorflow. keras. layers import Conv2D, Dense, Flatten, MaxPooling2D  # == Maxpool2D

model= Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28,28,1),
                 padding= 'same', # kernel_size=(3,3) 일 땐, 0으로 된 패딩이 양쪽으로 생김. 
                 strides=2,  # 보폭. 연산으로 2칸씩 넘어감. Maxpooling2D 랑은 다름. 1 = 1칸씩.
                 activation= 'relu'))    #(28,28,128)
model.add(MaxPooling2D())   # max_pooling2d (MaxPooling2D  (None, 14, 14, 128)      0)  / 영역이 겹치지 않는 2*2
#kernel_size 내의 특성 중 가장 높은 특성만 선택 >> 연산량이 반으로 줄어듦. 큰 데이터에서 사용하지만, 결과치 보고 사용여부 결정. 마지막에 남은 데이터는 연산 x 
model.add(Conv2D(filters=64, kernel_size=(2,2),
                 padding = 'same'
                 ))    #(14,14,64)    ###다음 레이어의 input_shape == input_shape - kernel_size +1
model.add(Conv2D(filters=64, kernel_size=(2,2)))    #(13,13,64)
model.add(Flatten())    #10816
model.add(Dense(32, activation= 'relu'))        #input_shape= (40000,)
                     #(60000,40000) 이 인풋.  (batch_size, input_dim)
model.add(Dense(10, activation= 'softmax'))

model.summary()

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
**기존성능
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 27, 27, 128)       640

 conv2d_1 (Conv2D)           (None, 26, 26, 64)        32832

 conv2d_2 (Conv2D)           (None, 25, 25, 64)        16448

 flatten (Flatten)           (None, 40000)             0

 dense (Dense)               (None, 32)                1280032

 dense_1 (Dense)             (None, 10)                330

=================================================================
Total params: 1,330,282
Trainable params: 1,330,282
Non-trainable params: 0
_________________________________________________________________
loss:  2.3012239933013916
acc:  0.11349999904632568

**padding 적용
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 128)       640

 conv2d_1 (Conv2D)           (None, 28, 28, 64)        32832

 conv2d_2 (Conv2D)           (None, 27, 27, 64)        16448

 flatten (Flatten)           (None, 46656)             0

 dense (Dense)               (None, 32)                1493024

 dense_1 (Dense)             (None, 10)                330

=================================================================
Total params: 1,543,274
Trainable params: 1,543,274
Non-trainable params: 0
_________________________________________________________________
loss:  0.1331486701965332
acc:  0.9674999713897705

**padding 적용, kernal_size = (3,3)
loss:  0.12937140464782715
acc:  0.97079998254776

**Maxpooing 적용
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 128)       1280

 max_pooling2d (MaxPooling2D  (None, 14, 14, 128)      0
 )

 conv2d_1 (Conv2D)           (None, 14, 14, 64)        32832

 conv2d_2 (Conv2D)           (None, 13, 13, 64)        16448

 flatten (Flatten)           (None, 10816)             0

 dense (Dense)               (None, 32)                346144

 dense_1 (Dense)             (None, 10)                330

=================================================================
Total params: 397,034
Trainable params: 397,034
Non-trainable params: 0

loss:  0.08170416951179504
acc:  0.9772999882698059

'''


#early, mcp, val 적용.
