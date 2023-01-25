from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
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

#2. 모델구성(순차형)
model= Sequential()
# model.add(Dense(5, input_dim=13))   #(506,13)>(13)
model.add(Dense(1, input_shape=(13,)))  
model.add(Dense(15))
model.add(Dense(5))
model.add(Dropout(0.3))
model.add(Dense(1))
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


mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name)  #훈련결과만 저장
                                                                                            #파일명에 시간, 날짜, val_loss 결과치 넣기
hist=model.fit(x_train, y_train, epochs=50, batch_size=1,                                      # k30_0112_1522_0047-19.2113.hdf5
        validation_split=0.2, callbacks=[earlystopping, mcp],  #반환값, 2개 이상은 리스트로
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
loss : 28.18191146850586
RMSE : 5.308663843249439
R2 : 0.7126610155918137
'''