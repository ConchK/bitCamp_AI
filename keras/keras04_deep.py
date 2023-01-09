import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,5,4])

#2. 모델구성

model=Sequential()
model.add(Dense(3, input_dim=1)) 
#Input-layer/ 첫번째 줄을 제외한 인풋은 명시하지 않음 
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(1)) 
#Ouyput-layer

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=250)

#4. 평가, 예측
results=model.predict([6])
print('6의 결과: ' ,results)

