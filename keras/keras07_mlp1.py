import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y=np.array([2,4,6,8,10,12,14,16,18,20])

#데이터.shape=행렬의 구조
print(x.shape)  #(2,10)
print(y.shape)  #(10,)

x=x.T  #T는 전치. 행과 열(데이터의 특성)을 바꿈
print(x.shape)  #(10, 2)

#2. 모델구성
model=Sequential()
model.add(Dense(3,input_dim=2))   #input_dim은 열의 갯수
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=180, batch_size=4)

#4. 평가, 예측
loss=model.evaluate(x,y)  #evaluate의 디폴트값 32/ 통상적으로 명시x
print('loss= ',loss)

result=model.predict([[10,1.4]])
print('[10,1.4의 예측값 : ',result)

'''
결과:
loss=  0.003985154442489147
[10,1.4의 예측값 :  [[20.008806]]
'''
