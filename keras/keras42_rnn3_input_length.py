import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN  #rnn 레이어
#SimpelRNN은 잘 사용하지 않음. ==VanillaRNN
#timestep을 길게 잡을 경우 ex)100일 앞의 연산 결과가 뒤에 미치지 못함.
#그래서 LSTM or GRU 사용

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
# model.add(SimpleRNN(units = 64, input_shape=(3,1), activation='relu'))   #[1,2,3]에서 하나식 연산 한다는 걸 명시해야 함(feature) >연산량이 많아짐.
model.add(SimpleRNN(units = 64, input_length= 3, input_dim= 1))  #위랑 동일. input_dim은 1차원에서 사용했음. 가독성 떨어짐.
model.add(Dense(44, activation= 'relu'))
model.add(Dense(22, activation= 'relu'))
model.add(Dense(1))     #flatten 필요없음.

model.summary()

'''파라미터 개수가 4224인 이유
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 64)                4224
                            batch를 명시 하지 않았기 때문에 None
 dense (Dense)               (None, 44)                2860

 dense_1 (Dense)             (None, 22)                990

 dense_2 (Dense)             (None, 1)                 23

=================================================================
Total params: 8,097
Trainable params: 8,097
Non-trainable params: 0
_________________________________________________________________

RNN Total params = recurrent_weights + input_weights + biases

= (num_units*num_units)+(num_features*num_units) + (1*num_units)

= (num_features + num_units)* num_units + num_units

결과적으로,

( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수)

를 참조하여 위의 total params 를 구하면

(64*64) + (1*64) + (1*64) = 4224

== units * ( featrue + bias + units ) = parameter

==> 어쨌든, 연산량이 많다.
'''