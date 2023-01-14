#[실습]- R2를 음수가 아닌 0.5이하로 줄이기
# 데이터는 건들지 말것.
# 레이어는 인풋 아웃풋 포함 7개 이상 -레이어가 너무 깊으면 기울기 때문에 초기 정보를 잊음
# batch_size=1
# 히든레이어의 노드는 각각 10개~100개
# train 70%
# epochs 100이상 
# loss지표는 mse or mae 
# activation 사용금지 -함정함수

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x=np.array(range(1,21))
y=np.array(range(1,21))

x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )

print('x_train :', x_train)  
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(80))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(60))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

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
RMSE : 0.18236157404390382
R2 : 0.998520139959506
'''