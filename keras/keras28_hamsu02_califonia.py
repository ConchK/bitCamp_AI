from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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


# #2. 모델구성
# model=Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dense(80))
# model.add(Dense(40))
# model.add(Dense(20))
# model.add(Dense(1))
# model.summary()
# #Total params: 5,051

#2. 모델구성(함수형)  
input1= Input(shape=(8,))
dense1= Dense(10)(input1)
dense2= Dense(80)(dense1)
dense3= Dense(40)(dense2)
dense4= Dense(20)(dense3)
output1= Dense(1)(dense4)
model= Model(inputs=input1, outputs=output1)   #정의가 마지막으로 
model.summary()
#Total params: 5,051





#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping= EarlyStopping(monitor='val_loss', mode=min, patience=10, restore_best_weights=True, verbose=1)
hist=model.fit(x_train, y_train, epochs=100, batch_size=1, 
          validation_split=0.2, callbacks=[earlystopping], verbose=1)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict=model.predict(x_test)

# print(y_test)
# print(y_predict)


from sklearn.metrics import mean_squared_error,r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

print('RMSE :', RMSE(y_test, y_predict))

r2=r2_score(y_test, y_predict)    
print("R2 :", r2)

'''
loss : [0.5994833707809448, 0.6061038970947266]
RMSE : 0.7742631114464298
R2 : 0.5466321466828501
'''


