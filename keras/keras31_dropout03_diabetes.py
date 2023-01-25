from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.datasets import load_diabetes

#1. 데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target
'''
print(x)
print(x.shape)  #(442, 10)
print(y)
print(y.shape)  #(442,)

print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )   

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()

scaler.fit(x_train)
x_train= scaler.transform(x_train)
# x_train= scaler.fit_transform(x_train)    #x범위 만큼의 가중치 생성

x_test= scaler.transform(x_test)

# #2. 모델구성
# model=Sequential()
# model.add(Dense(5, input_dim=10, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(60, activation= 'sigmoid'))
# model.add(Dense(40, activation= 'relu'))
# model.add(Dense(20, activation= 'relu'))
# model.add(Dense(1, activation= 'linear'))
# model.summary()
# #Total params: 3,696

#2. 모델구성(함수형)  
input1= Input(shape=(10,))
dense1= Dense(5, activation= 'relu')(input1)
drop1= Dropout(0.5)(dense1)
dense2= Dense(60, activation= 'sigmoid')(drop1)
dense3= Dense(40, activation= 'relu')(dense2)
dense4= Dense(20, activation= 'relu')(dense3)
output1= Dense(1, activation= 'linear')(dense4)
model= Model(inputs=input1, outputs=output1)   #정의가 마지막으로 
model.summary()
#Total params: 3,696

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k31_3_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss', mode=min, patience=10, restore_best_weights=True, verbose=1)
mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name)  #훈련결과만 저장

hist=model.fit(x_train, y_train, epochs=50, batch_size=1,
         validation_split=0.2, callbacks= [earlystopping, mcp], verbose=1)

#4. 평가, 예측

loss=model.evaluate(x_test,y_test)  #x_test,y_test 값으로 평가
print('loss :', loss)
y_predict=model.predict(x_test)  #x_test값으로 y_predict 예측
# print('x_test :', x_test)
# print('y_predict :', y_predict)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))
r2=r2_score(y_test, y_predict)
print('R2: ', r2)

'''
loss : [3514.980712890625, 50.122772216796875]
RMSE:  59.28727249468124
R2:  0.40892006051354823
'''

