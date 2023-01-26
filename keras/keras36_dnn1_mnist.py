import numpy as np
from tensorflow.keras.datasets import mnist

#dnn은 다차원 가능

#1.데이터

(x_train, y_train), (x_test, y_test)= mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

#이미지는 3차원 > 4차원으로 바꿔준다.
# x_train= x_train.reshape(60000,28,28,1) #60000 행, (28,28,1)열
# x_test= x_test.reshape(10000,28,28,1)

# 3차원에서 2차원으로 바꿔준다. >> Flatten 형태
x_train= x_train.reshape(60000,28*28)
x_test= x_test.reshape(10000,28*28)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)




print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))


#2 모델

from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout 

model= Sequential()                #28*28
model.add(Dense(500, input_shape=(784,), activation= 'relu'))  
model.add(Dropout(0.3))  
model.add(Dense(400, activation= 'relu'))    
model.add(Dropout(0.3))  
model.add(Dense(300, activation= 'relu'))   
model.add(Dense(100, activation= 'relu'))      
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

**CNN
loss:  0.08170416951179504
acc:  0.9772999882698059

**DNN
loss:  0.07960204780101776
acc:  0.9765999913215637
'''
