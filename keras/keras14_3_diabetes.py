# [과제, 실습]
# R2 0.62 이상

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

#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=5)

#4. 평가, 예측

loss=model.evaluate(x_test,y_test)  #x_test,y_test 값으로 평가
print('loss :', loss)
y_predict=model.predict(x_test)  #x_test값으로 y_predict 예측
print('x_test :', x_test)
print('y_predict :', y_predict)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))
r2=r2_score(y_test, y_predict)
print('R2: ', r2)

'''
RMSE:  54.341179399509045
R2:  0.5034289486936249
'''