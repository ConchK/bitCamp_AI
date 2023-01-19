from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x= np.array([1,2,3])
y= np.array([1,2,3])

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

model.summary()    #모델의  archtecture 모형 노드 사이의 연산- 파라미터
'''
Model: "sequential"   다른 사람이 훈련시킨 결과 + 내가 훈련시킨 결과 = 개꿀
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 5)                 10  -> (input + bias) *output =(1+1)*5

 dense_1 (Dense)             (None, 15)                90   -> (5+1)*15

 dense_2 (Dense)             (None, 30)                480   -> (15+1)*30

 dense_3 (Dense)             (None, 40)                1240

 dense_4 (Dense)             (None, 30)                1230

 dense_5 (Dense)             (None, 10)                310

 dense_6 (Dense)             (None, 1)                 11

=================================================================
Total params: 3,371
Trainable params: 3,371
Non-trainable params: 0
'''















