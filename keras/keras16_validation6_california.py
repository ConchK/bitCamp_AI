# [실습]
# R2 0.55 이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(120))
model.add(Dense(160))
model.add(Dense(200))
model.add(Dense(120))
model.add(Dense(40))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=300, batch_size=10,
          validation_split=0.2)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict=model.predict(x_test)

print(y_test)
print(y_predict)


from sklearn.metrics import mean_squared_error,r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

print('RMSE :', RMSE(y_test, y_predict))

r2=r2_score(y_test, y_predict)    
print("R2 :", r2)

'''
RMSE : 0.8443698548987285
R2 : 0.4608134803574575
'''