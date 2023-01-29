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
model=Sequential()          
model.add(LSTM(units= 5, input_shape=(3,1), activation='relu'))   #simplernn이랑 input, output 동일.
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

y_pred = np.array([50,60,70]).reshape(1,3,1)   #input_shape = (3,1) 때문에 3차원으로 변형 시켜야 함.
result = model.predict(y_pred)
print('[50,60,70]의 결과 : ', result)   #80을 구하시오.

'''
loss :  [24.711551666259766, 0.0]
[50,60,70]의 결과 :  [[101.21942]]
'''