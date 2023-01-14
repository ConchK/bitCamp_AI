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


from sklearn.metrics import mean_squared_error,r2_score  #MSE와 r2 동시에 import
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
print('RMSE :', RMSE(y_test, y_predict))

#회귀 분석 모델(결정계수), max값 1에 가까울 수록 설명력(정확도)이 높음
r2=r2_score(y_test, y_predict)    
print("R2 :", r2)

'''
RMSE : 3.9167981485875965
R2 : 0.6359346878549426

회귀모델이 주어진 자료에 얼마나 적합한지를 평가하는 지표
y의 변동량대비 모델 예측값의 변동량을 의미함
0~1의 값을 가지며, 상관관계가 높을수록 1에 가까워짐
r2=0.3인 경우 약 30% 정도의 설명력을 가진다 라고 해석할 수 있음
sklearn의 r2_score의 경우 데이터가 arbitrarily할 경우 음수가 나올수 있음
음수가 나올경우 모두 일괄 평균으로 예측하는 것보다 모델의 성능이 떨어진다는 의미
결정계수는 독립변수가 많아질 수록 값이 커지기때문에, 독립변수가 2개 이상일 경우 조정된 결정계수를 사용해야 함
 

'''
