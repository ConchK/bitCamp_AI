from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x=np.array(range(1,21))
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )

print('x_train :', x_train)  
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(15))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict=model.predict(x_test)

print(y_test)
print(y_predict)

#RMSE 함수는 아직 없어서 직접 만들어 사용. -회귀 분석 모델
from sklearn.metrics import mean_squared_error  #MSE함수 불러옴.
def RMSE(y_test, y_predict):  #y_test,y_predict로 만든 RMSE 함수는 이거임 ㅇㅇ 
    return np.sqrt(mean_squared_error(y_test, y_predict))   #y_test, y_predict로 만든 MSE에 루트(sqrt)를 씌워서 내놔라.

print('RMSE :', RMSE(y_test, y_predict))   #값이 낮을수록 정밀도가 높음 
# RMSE : 3.8738913732818863