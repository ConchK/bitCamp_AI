import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([range(10), range(21, 31), range(201, 211)])  #0부터 10개의 수=10개의 데이타 / 21~30 / 201~210 ,마지막 수-1
# print(range(10))
print(x.shape)  #(3, 10)
y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])

x=x.T
y=y.T
print(x.shape, y.shape)   #(10, 3) (10, 2)

model=Sequential()
model.add(Dense(5,input_dim=3))  #x열의 갯수=input_dim
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(35))
model.add(Dense(45))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(2))  #y열의 갯수

model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=200, batch_size=3)

loss=model.evaluate(x,y)
print('loss= ',loss)

result=model.predict([[9,30,210]])
print('9,30,210의 예측값 :',result)

'''
결과 :
loss=  0.19912490248680115
9,30,210의 예측값 : [[9.65905   1.7763605]]
'''
