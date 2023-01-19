import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array(range(1,17))
y=np.array(range(1,17))
#넘파이 리스트 슬라이싱 7:3
x_train=x[:11]  #=[:7]
y_train=y[:11]  #=[:7]
x_test=x[10:13]   #=[7:]
y_test=y[10:13]   #=[7:]
x_validation=x[13:]  #=[:7]
y_validation=y[13:]   #=[7:]

# x_train=np.array(range(1,11))
# y_train=np.array(range(1,11))
# x_test=np.array([11,12,13])
# y_test=np.array([11,12,13])
# x_validation=np.array([14,15,16])
# y_validation=np.array([14,15,16])

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10,
          validation_data=(x_validation, y_validation))  #훈련, 검증 반복  

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)  #x_test,y_test 값으로 평가
print('loss :', loss)
result=model.predict([17])  #x_test값으로 y_predict 예측
print('17의 예측값 :', result)

'''
loss : 0.08083165436983109
17의 예측값 : [[16.374092]]
'''