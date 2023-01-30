import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],
            [4,5,6],[5,6,7],[6,7,8],
            [7,8,9],[8,9,10],[9,10,11],
            [10,11,12],[20,30,40],
            [30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape)     #(13, 3) (13,)

x = x.reshape(13,3,1)
print(x.shape)

#2. 모델구성
model=Sequential()          #(N,3,1)           
model.add(LSTM(units= 64, input_shape=(3,1), return_sequences=True ,activation='relu'))   # >> 3차원을 input 하면 2차원 (N, 64)로 output 됨. >> Flatten 필요 없음. 
#  LSTM을 두 개 이상 쓰려면 return_sequences= True > input 차원을 유지 / lstm은 굳이 여러번 쓸 필요 없음. 
model.add(LSTM(units= 32, activation='relu'))    # (None, 3, 64) input 해야함.
model.add(Dense(60, activation= 'relu'))               
model.add(Dense(40, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(1))     #flatten 필요없음.

model.summary()

#3. 컴파일, 훈련         
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

model.fit(x, y, epochs=100, batch_size=1, 
            validation_split=0.1, verbose=2)

#4. 평가, 예측
loss= model.evaluate(x,y)
print('loss : ', loss)

y_pred = np.array([50,60,70]).reshape(1,3,1)   # input_shape = (3,1) 때문에 3차원으로 변형 시켜야 함. >>차원을 동일하게.
result = model.predict(y_pred)
print('[50,60,70]의 결과 : ', result)   #80을 구하시오.

'''
loss :  [24.711551666259766, 0.0]
[50,60,70]의 결과 :  [[101.21942]]
'''