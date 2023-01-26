from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
import numpy as np
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target
'''
print(x)
print(x.shape)  #(20640, 8)
print(y)
print(y.shape)  #(20640,)

print(datasets.feature_names)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)   #DESCR=discribe
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

print(x_train.shape, x_test.shape)      #(14447, 8) (6193, 8)

x_train= x_train.reshape(14447,2,2,2)
x_test= x_test.reshape(6193,2,2,2)


#2. 모델구성
# model=Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dropout(0.5))
# model.add(Dense(80))
# model.add(Dense(40))
# model.add(Dense(20))
# model.add(Dense(1))
# model.summary()
# #Total params: 5,051

model= Sequential()                      #4차원으로 변형
model.add(Conv2D(40, (2,2), input_shape= (2,2,2), padding = 'same', activation= 'relu'))
model.add(Conv2D(40, (2,2), padding = 'same', activation= 'relu'))
model.add(Flatten())
model.add(Dense(80, input_shape=(8,), activation= 'relu'))  
model.add(Dense(40,activation= 'relu'))
model.add(Dense(1, activation= 'softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k31_2_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss', mode=min, patience=10, restore_best_weights=True, verbose=1)
# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name)  #훈련결과만 저장

hist=model.fit(x_train, y_train, epochs=100, batch_size=1, 
          validation_split=0.2, callbacks=[earlystopping], verbose=1)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict=model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

print('RMSE :', RMSE(y_test, y_predict))

r2=r2_score(y_test, y_predict)    
print("R2 :", r2)

'''
loss : [2.4309988021850586, 1.1396371126174927]
RMSE : 1.559166039555551
R2 : -0.8384775002063716
'''


