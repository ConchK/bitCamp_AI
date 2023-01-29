import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN  #rnn 레이어

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])  #(10, )  데이터 하나를 가지고 훈련.
#시계열 데이터는 y값이 없다.
#3일치의 데이터로 다음날의 주가를 예상

x = np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6],
            [5,6,7],
            [6,7,8],
            [7,8,9]])   # y= 11 없기 때문에 [8,9,10] 하지 않음.

y = np.array([4,5,6,7,8,9,10])    #3일치 데이터의 결과들

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(7,3,1)        #[1,2,3],[2,3,4],... -> [[1],[2],[3]],[[2],[3],[4]],... 형태가 바뀜. 1다음 2, 2다음 3,...임을 훈련시킴.
print(x.shape)  #(7, 3, 1) 3차원. 7개의 데이터를 3개씩 잘라 1개씩 연산.

#2. 모델구성
model=Sequential()            #행7 빼고 열
model.add(SimpleRNN(units = 64, input_shape=(3,1), activation='relu'))   #[1,2,3]에서 하나식 연산 한다는 걸 명시해야 함 >연산량이 많아짐.
                   #units은 신경망에 존재하는 뉴런의 개수
model.add(Dense(44, activation= 'relu'))
model.add(Dense(22, activation= 'relu'))
model.add(Dense(1))     #flatten 필요없음.

#3. 컴파일, 훈련         
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

model.fit(x, y, epochs=100, batch_size=1, 
            validation_split=0.1, verbose=2)

#4. 평가, 예측
loss= model.evaluate(x,y)
print('loss : ', loss)

y_pred = np.array([8,9,10]).reshape(1,3,1)   #input_shape = (3,1) 때문에 3차원으로 변형 시켜야 함.
result = model.predict(y_pred)
print('[8,9,10]의 결과 : ', result)

'''
loss :  [0.001316992798820138, 0.0]
[8,9,10]의 결과 :  [[11.187891]]
'''