import tensorflow as tf   
print(tf.__version__)
import numpy as np

#1. 데이터
x=np.array([1,2,3])
y=np.array([1,2,3])

#2. 모델구성 
from tensorflow.keras.models import Sequential  #레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense   #뉴런의 입력input과 출력output을 연결

model=Sequential()
model.add(Dense(1, input_dim=1))
# y줄의(output) 123이 1, x줄의(input) 123이 input_dim=1의 1 /Dense=(y=yx+b)를 1번 계산. /dim=dimention 

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
#loss값 최소화 위해 mae를 쓰겠다는 뜻, loss를 최적화 하기 위해 adam을 사용
model.fit(x, y, epochs=15)
#훈련시작. epochs:훈련수

#4. 평가, 예측
result = model.predict([4])
print('결과 : ',result)

'''
결과 :  [[-6.7709684]]
'''