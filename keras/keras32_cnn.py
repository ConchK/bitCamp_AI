from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten   # Conv2 :2차원 평면(이미지) , Conv1D :1차원 
# CNN은 컴퓨터 비젼 분야에 주로 활용 (그외에 음성, NLP 등등) / dnn도 가능
# 행은 데이터의 갯수.
# 3차원(가로, 세로, 칼라) 필요
# Convolutional neural network

model= Sequential()

model.add(Conv2D(filters=10,        #4*4로 바뀐 수치를 10장으로 만들겠다 > 연산량이(특성이 증가한 데이터) 늘어남. >>output
                kernel_size=(2,2),  #가로 2칸, 세로2칸 지정.
                padding = 'same', #  특성의 크기는 줄어들지 않음. 디폴트는 'valid'
                #연산 중 가장 자리에 있는 데이터의 소멸을 막기 위해, 특성을 넓게 잡기 위해 끝에 0으로 된 패딩을 한 줄씩 넣어준다. >> 특성은 높이고 데이터는 늘리고. 
                # (5,5,1) 필터 100 > (5,5,100)
                input_shape=(5,5,1))) # =흑백그림 1장: 가로 5칸, 세로 5칸, 흑백 1(5,5,1)  >>kernel_size=(2,2)로 계산 후 4*4로 바뀜 :정사각형 4칸이 1칸으로(4,4,10)
              #(batch_size, rows, colums, channels)  
model.add(Conv2D(5, kernel_size=(2,2)))   #(3,3,5)  순차 레이어는 상위 레이어의 아웃풋이 하위 레이어의 인풋이기 때문에 따로 명시 하지 않음.
#모든 것의 사용 여부는 최종 결과치를 보고 결정.

model.add(Flatten())    #한 줄로 펼쳐 열로 만듦. 3*3*5==45  (45,)
model.add(Dense(units=10))        #(N, 4, 4,10)
          #인풋은 (batch_size, input_dim)
model.add(Dense(4, activation= 'relu'))   #


model.summary()

'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50 == kernael_size 2*2 * input 열 1 * filters(output) 10 + bias(output이랑 동일) 10
                        데이터의 수는 상관 없음.==None
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205 == kernael_size 2*2 * input 열(=앞에서의 output) 10 * filters(output) 5 + bias(output이랑 동일) 5
                         
 flatten (Flatten)           (None, 45)                0

 dense (Dense)               (None, 10)                460 == 45*10+10

 dense_1 (Dense)             (None, 4)                 44 ==10*4+4

=================================================================
Total params: 726
Trainable params: 726
Non-trainable params: 0
_________________________________________________________________
3 x 3 (필터 크기) x 32 (#입력 채널) x 64(#출력 채널) + 64 = 18496


Input
: 모양이 있는 4+D 텐서: batch_shape + (channels, rows, cols)if data_format='channels_first' 
  또는 모양이 있는 4+D 텐서: batch_shape + (rows, cols, channels)if data_format='channels_last'.

Output
: 모양이 있는 4+D 텐서: batch_shape + (filters, new_rows, new_cols)if data_format='channels_first'
  또는 모양이 있는 4+D 텐서: batch_shape + (new_rows, new_cols, filters)if data_format='channels_last'. 
  rows 패딩 으로 cols인해 값이 변경되었을 수 있습니다.

filters 
: 정수, 출력 공간의 차원(예: 컨볼루션의 출력 필터 수).



'''