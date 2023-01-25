from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
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

#2. 모델구성(함수형)  
input1= Input(shape=(13,))
dense1= Dense(1)(input1)
dense2= Dense(15)(dense1)
dense3= Dense(5)(dense2)
output1= Dense(1)(dense3)
model= Model(inputs=input1, outputs=output1)   #정의가 마지막으로 
model.summary()
# Total params: 130

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=False, verbose=1)  

import datetime    #datetime이라는 데이타 타입
date= datetime.datetime.now()
print(date)   #2023-01-12 14:58:53.272536
print(type(date))       #<class 'datetime.datetime'>  >>데이타(숫자) 형태에서 문자형으로 바꿔야 함.
date= date.strftime("%m%d_%H%M")     #0112_1458
print(date)
print(type(date))       #<class 'str'>

filepath= './_save/MCP/'
filename= '{epoch:04d}-{val_loss:.4f}.hdf5'      #epoch 번호(정수) 4자리 수-val_loss 값 소수점 4자리 까지


mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=filepath +'k30_'+date+'_'+filename)  #훈련결과만 저장
                                                                                            #파일명에 시간, 날짜, val_loss 결과치 넣기
hist=model.fit(x_train, y_train, epochs=50, batch_size=1,                                      # k30_0112_1522_0047-19.2113.hdf5
        validation_split=0.2, callbacks=[earlystopping, mcp],  #반환값, 2개 이상은 리스트로
        verbose=1)



#4. 평가, 예측
print("================1. 기본 출력==========================") 
loss=model.evaluate(x_test, y_test)
print('loss :', loss)
