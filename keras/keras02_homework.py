from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성 (Sequential은 레이어에 순차적으로 연산)
model=Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=30)

#4. 평가, 예측
# [13] : 예측해봐요
result = model.predict([13])
print('결과 : ',result)

'''
결과 :  [[10.7677555]]
'''
