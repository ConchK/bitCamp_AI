import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array(range(1,17))
y=np.array(range(1,17))

#10:3:3 train_test_split 사용해서 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.85, shuffle=True, random_state=123
        )
print(x_train)   #[ 1  6 10  9 12  4  2  7 16 13  3 14 15]
print(y_train)     #[ 1  6 10  9 12  4  2  7 16 13  3 14 15]

#train 에서 validation을 공유
x_train, x_validation, y_train, y_validation= train_test_split(x_train,y_train,
        train_size=0.8, shuffle=True, random_state=123
        )
print(x_validation)   #[ 7  3 12]
print(y_validation)    #[ 7  3 12]


'''
#넘파이 리스트 슬라이싱 7:3
x_train=x[:11]  
y_train=y[:11]  
x_test=x[10:13]   
y_test=y[10:13]  
x_validation=x[13:]  
y_validation=y[13:]  

# x_train=np.array(range(1,11))
# y_train=np.array(range(1,11))
# x_test=np.array([11,12,13])
# y_test=np.array([11,12,13])
# x_validation=np.array([14,15,16])
# y_validation=np.array([14,15,16])
'''

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
loss : 68.41297149658203
17의 예측값 : [[0.09981167]]
'''