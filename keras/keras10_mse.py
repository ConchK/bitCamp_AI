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
model.compile(loss='mse', optimizer='adam')
#loss가 0이 되면 안되기 때문에 mse를 사용> mse가 너무 크면 루트를 씌워서 rmse> rmse도 크면 log를 씌워서 rmsle 참조(https://steadiness-193.tistory.com/277)
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss :', loss)