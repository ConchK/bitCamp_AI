import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
#from tensorflow > keras > models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,5,4,6])

#2. 모델구성

model=Sequential()
model.add(Dense(3, input_dim=1)) 
#Input-layer/ 첫번째 줄을 제외한 인풋은 명시하지 않음 
model.add(Dense(5))
model.add(Dense(4)) 
model.add(Dense(2))
model.add(Dense(1)) 
#Ouyput-layer

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=30, batch_size=2)
#fit에서 훈련 후 가중치 생성.(y=wx+b)/ 훈련 횟수, 배치 사이즈 조절/ 배치 사이즈 명시 없을 시 디폴트 값 32

#4. 평가, 예측
loss=model.evaluate(x,y)  #loss 값으로 평가, 반환
print('loss : ', loss)
results=model.predict([6])
print('6의 결과: ' ,results)

#loss 수치가 낮으면 가중치에 최적화. 판단의 기준.

"""
블럭주석


"""