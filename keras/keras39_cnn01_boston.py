#31_1

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
path= './_save/'

#1. 데이터
datasets= load_boston()
x=datasets.data
y=datasets.target
# print(x.shape, y.shape)   #(506, 13) (506,)



x_train, x_test, y_train, y_test= train_test_split(
        x,y, shuffle=True, random_state=333, test_size=0.2
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()

scaler.fit(x_train)
x_train= scaler.transform(x_train)
# == x_train= scaler.fit_transform(x_train)   
x_test= scaler.transform(x_test)

print(x_train.shape, x_test.shape)   # #(404, 13) (102, 13)  내용물, 순서가 바뀌지 않도록 reshape >> (13,1,1)=input_shape

x_train= x_train.reshape(404,13,1,1)
x_test= x_test.reshape(102,13,1,1)



#2. 모델구성(순차형)
model= Sequential()
# model.add(Dense(5, input_dim=13))   #(506,13)>(13)   kernel_size를 직사각형 형태로 바꿔줌. 
model.add(Conv2D(40, (2,1), input_shape= (13,1,1), padding = 'same', activation= 'relu'))
model.add(Flatten())
model.add(Dense(100, input_shape=(13,), activation= 'relu'))  
model.add(Dense(60,activation= 'relu'))
model.add(Dense(40,activation= 'relu'))
model.add(Dense(1, activation= 'softmax'))
model.summary()
# Total params: 130 



# #2. 모델구성(함수형)  
# input1= Input(shape=(13,))
# dense1= Dense(1)(input1)
# dense2= Dense(15)(drop1)
# dense3= Dense(5)(dense2)
# drop2= Dropout(0.3)(dense3)
# output1= Dense(1)(drop2)
# model= Model(inputs=input1, outputs=output1)   #정의가 마지막으로 
# model.summary()
# # Total params: 130

#3. 컴파일, 훈련   - dropout은 훈련 때만
model.compile(loss='mse', optimizer='adam')    

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)  

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k31_1_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5' 


# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name)  #훈련결과만 저장
                                                                                            #파일명에 시간, 날짜, val_loss 결과치 넣기
hist=model.fit(x_train, y_train, epochs=100, batch_size=1,                                      # k30_0112_1522_0047-19.2113.hdf5
        validation_split=0.2, callbacks=[earlystopping],  #반환값, 2개 이상은 리스트로
        verbose=1)



#4. 평가, 예측  - 평가할 때는 dropout 안 하고
loss=model.evaluate(x_test, y_test)
print('loss :', loss)

from sklearn.metrics import mean_squared_error,r2_score
y_predict=model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   
print('RMSE :', RMSE(y_test, y_predict))

r2=r2_score(y_test, y_predict)    
print("R2 :", r2)

'''
oss : 584.281494140625
RMSE : 24.17191491355692
R2 : -4.957255332962484
'''