from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=10, activation='relu'))
model.add(Dense(60, activation= 'sigmoid'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(1, activation= 'linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping= EarlyStopping(monitor='val_loss', mode=min, patience=10, restore_best_weights=True, verbose=1)
hist=model.fit(x_train, y_train, epochs=200, batch_size=5,
         validation_split=0.1, callbacks= [earlystopping], verbose=1)

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
RMSE:  58.16238228476438
R2:  0.4311370474461458

스케일 변환 후
loss : [3245.838623046875, 45.575531005859375]
RMSE:  56.97226122215378
R2:  0.454179061643039
'''

