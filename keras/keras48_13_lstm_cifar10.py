from tensorflow.keras.datasets import cifar10
import numpy as np
#1.데이터

(x_train, y_train), (x_test, y_test)= cifar10.load_data()

print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     #(10000, 32, 32, 3) (10000, 1)  


# 4차원에서 2차원으로 바꿔준다. 
x_train= x_train.reshape(50000,32*32*3)
x_test= x_test.reshape(10000,32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

# 2차원에서 3차원으로 바꿔준다. 
x_train= x_train.reshape(50000,32*32,3)
x_test= x_test.reshape(10000,32*32,3)


#2.모델

from tensorflow. keras. models import Sequential
from tensorflow. keras. layers import Dense, Dropout, LSTM

model= Sequential()               
model.add(LSTM(units = 40, input_shape=(32*32,3), activation= 'relu'))  
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

'''
***cnn
loss:  2.3025941848754883
acc:  0.10000000149011612

**dnn
loss:  1.9722217321395874
acc:  0.2676999866962433
'''