import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array(range(10))
y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
            [9,8,7,6,5,4,3,2,1,0]])

y=y.T
print('x.shape, y.shape: ',x.shape, y.shape)  # (10,) (10, 3)

model=Sequential()
model.add(Dense(5,input_dim=1))  #x열의 갯수=input_dim
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(35))
model.add(Dense(45))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(3))  #y열의 갯수

model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=200, batch_size=3)

loss=model.evaluate(x,y) #
print('loss= ',loss)

result=model.predict([9])
print('9의 예측값 :',result)

'''
결과 :
loss=  0.20552575588226318
9의 예측값 : [[10.175534   1.4578452 -0.487389 ]]
'''
