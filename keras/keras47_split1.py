#주어진 split 함수로 data 자르기.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


a = np.array(range(1,11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []        #빈 리스트를 만들어 줌.
    for i in range(len(dataset) - timesteps + 1):       #데이터의 길이 10 - timestep 5 + 1 만큼 반복(총량) =3번
        subset = dataset[i : (i + timesteps)]   #a 의 [0번째 : 0 + timestep 5] 이 subset에 들어감. > 첫번째 리스트
        aaa.append(subset)  #빈 리스트에 넣어줌.
    return np.array(aaa) 

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) #(6, 4) (6,)

x = x.reshape(6,4,1)
print(x.shape)

#LSTM 모델 구성

#2. 모델구성
model = Sequential()
model.add(LSTM(units= 64, input_shape=(4,1), return_sequences=True ,activation='relu'))   # >> (N, 64)로 output 됨. 처음엔 2차원으로 넣기 때문에 flatten이 필요 없었음.
model.add(LSTM(units= 32, activation='relu'))   # 두번째에는 3차원을 넣어야 함.  <- (None, 3, 64) input
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

y_predict = np.array([7,8,9,10]).reshape(1,4,1)  
result = model.predict(y_predict)
print('[7,8,9,10]의 결과 : ', result)   #11을 구하시오.

'''
loss :  [0.007419023662805557, 0.0]
[7,8,9,10]의 결과 :  [[10.675914]]
'''