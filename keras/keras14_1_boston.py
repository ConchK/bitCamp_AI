# [실습]
# 1. train 0.7이상
# 2. R2 : 0.8이상 / RMSE사용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston  #sklearn에서 제공하는 load_boston 가져오기

#1. 데이터
dataset= load_boston()
x=dataset.data   #집에 대한 데이타
y=dataset.target   #집 값


print(x)
print(x.shape)  #(506집 수, 13조건)
print(y)
print(y.shape)  #(506,) 각 집 가격
'''
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']
print(dataset.DESCR)   #DESCR=discribe
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )   

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(80))
model.add(Dense(65))
model.add(Dense(40))
model.add(Dense(22))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=80, batch_size=1)  #batch_size 디폴트 32

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
RMSE : 5.842083977968378
R2 : 0.5777464925938227
# 평가지표: R2, RMSE
'''
