import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array(range(1,17))
y=np.array(range(1,17))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.85, shuffle=True, random_state=123
        )
print(x_train.shape, x_train)    #(13,)[ 1  6 10  9 12  4  2  7 16 13  3 14 15]
print(y_train.shape, y_train)     #(13,)[ 1  6 10  9 12  4  2  7 16 13  3 14 15]


#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10,
          validation_split=0.25)  #train 데이터의 25%를 훈련에 사용   > val_loss로 나옴. 판단의 기준

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)  #x_test,y_test 값으로 평가
print('loss :', loss)
result=model.predict([17])  #x_test값으로 y_predict 예측
print('17의 예측값 :', result)

'''
loss : 29.879182815551758
17의 예측값 : [[5.7358723]]

'''
