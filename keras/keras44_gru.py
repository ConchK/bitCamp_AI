import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
#SimpelRNN은 잘 사용하지 않음. ==VanillaRNN
#timestep을 길게 잡을 경우 ex)100일 앞의 연산 결과가 뒤에 미치지 못함.
#그래서 LSTM 사용. GRU.

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
#3차원 형태로만 바꿔주면 됨. dnn 2차원, cnn 4차원.

#2. 모델구성
model=Sequential()            #행7(앞에 none = batch) 빼고 열 ==(input_length=3(timesteps), input_dim=1(feature))
# model.add(SimpleRNN(units = 10, input_shape=(3,1), activation='relu'))   #[1,2,3]에서 하나식 연산 한다는 걸 명시해야 함(feature) >연산량이 많아짐.
#model.add(SimpleRNN(units = 64, input_length= 3, input_dim= 1))  #위랑 동일. input_dim은 1차원에서 사용했음. 가독성 떨어짐.
# model.add(LSTM(units= 10, input_shape=(3,1), activation='relu'))   #simplernn이랑 input, output 동일.
model.add(GRU(units= 10, input_shape=(3,1), activation='relu'))   #simplernn이랑 input, output 동일.


model.add(Dense(5, activation= 'relu'))
model.add(Dense(1))     #flatten 필요없음.

model.summary()

'''
SimpleRNN
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                120

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

=================================================================
Total params: 181
Trainable params: 181
Non-trainable params: 0
_________________________________________________________________

LSTM
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

=================================================================
Total params: 541
Trainable params: 541
Non-trainable params: 0
_________________________________________________________________

LSTM params = 4 * ((size_of_input_dim + 1) * size_of_output + size_of_output^2)
4 * (2 * 10 + 10^2) = 480

3개의 gate와 1개의 state 때문에 Simplernn 보다 4배 더 연산. 4배 더 좋아지는 건 아니지만 시간은 4배 더 걸림.



GRU
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 gru (GRU)                   (None, 10)                390

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

=================================================================
Total params: 451
Trainable params: 451
Non-trainable params: 0
_________________________________________________________________

2개의 gate. update, reset gate. / cell state와 hidden state 가 합해져 hidden state 1개.

'''