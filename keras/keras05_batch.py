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
model.add(Dense(5))
model.add(Dense(4)) 
model.add(Dense(2))
model.add(Dense(1)) 
#Ouyput-layer

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=15, batch_size=2)
#fit에서 배치 사이즈 조절/ 배치 사이즈 명시 없을 시 디폴트 값 32/ 
#6개의 데이터를 2개씩 잘라 작업./ batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서

#4. 평가, 예측
results=model.predict([6])
print('6의 결과: ' ,results)