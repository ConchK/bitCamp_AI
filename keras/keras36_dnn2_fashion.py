import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

# 3차원에서 2차원으로 바꿔준다. >> Flatten 형태
x_train= x_train.reshape(60000,28*28)
x_test= x_test.reshape(10000,28*28)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)


print(x_train[0])
print(y_train[0])

#2.모델

from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

model= Sequential()               #32*32*3
model.add(Dense(500, input_shape=(3072,), activation= 'relu'))  
model.add(Dropout(0.5))
model.add(Dense(400, activation= 'relu'))  
model.add(Dropout(0.3))
model.add(Dense(300, activation='relu'))  
model.add(Dense(200, activation= 'relu'))    
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

import matplotlib.pyplot as plt
plt.imshow(x_train[0], 'gray')    #밝은 부분이 데이터가 높음
plt.show()


