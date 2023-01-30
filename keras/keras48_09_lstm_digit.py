from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout,LSTM
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


#1. 데이타
datasets= load_digits()
x= datasets.data
y= datasets.target
# print(x.shape, y.shape)   #(1797, 64) (1797,)
# print(np.unique(y, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
# array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))  /1797개의 행

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])   #images[3]에 들어있는 숫자 '3'이 흑백의 이미지로 나옴. 한 칸에 0~255의 숫자가 들어감. 진한 곳엔 더 높은 숫자. 8칸*8칸==64칸의 열
# plt.show()                        

from tensorflow.keras.utils import to_categorical    
y= to_categorical(y)
# print(y)
# print(y.shape)   #(1797, 10)

x_train, x_test, y_train, y_test= train_test_split(
    x, y, shuffle=True,  #False의 문제점은 shuffle이 되지 않음. 
                        #True의 문제점은 특정 class를 제외할 수 없음.-데이터를 수집하다 보면 균형이 안맞는 경우.
                        #회귀는 데이터가 수치라서 상관 없음.
    stratify=y,     # y : yes  / 수치가 한 쪽으로 치우치는 걸 방지. y의 데이터가 분류형일 때만 가능.
    random_state=333,
    test_size=0.2
)
# print(y_train)
# print(y_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()

scaler.fit(x_train)
x_train= scaler.transform(x_train)
# x_train= scaler.fit_transform(x_train)    #x범위 만큼의 가중치 생성

x_test= scaler.transform(x_test)

print(x_train.shape, x_test.shape)      #(1437, 64) (360, 64)

x_train= x_train.reshape(1437,8,8)
x_test= x_test.reshape(360,8,8)


#2. 모델구성

model= Sequential()
model.add(LSTM(units = 40, input_shape= (8,8), activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation= 'relu'))  
model.add(Dropout(0.3))
model.add(Dense(40,activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#3. 컴파일, 훈련         
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k31_9_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  
# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name)  #훈련결과만 저장

model.fit(x_train, y_train, epochs=200, batch_size=5, 
            validation_split=0.2, verbose=2, callbacks=[earlystopping])

#4. 평가, 예측
loss, accuracy= model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict= model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)    #가장 큰 위치값을 찾아냄
print("y_pred(예측값) : ",y_predict)

# y_test= np.argmax(y_test, axis=1)
# print("y_test(원래값) : ",y_test)  

# acc= accuracy_score(y_test, y_predict)
# print("accuracy_score : ",acc)

'''
dnn
loss :  1.3346225023269653
accuracy :  0.4694444537162781
y_pred(예측값) :  [2 6 2 7 2 6 9 7 2 9 0 3 7 7 9 1 7 5 5 2 9 9 7 2 6 5 2 6 6 4 2 6 5 4 6 6 9
 9 0 9 1 5 6 3 0 6 4 2 9 5 5 7 6 9 9 5 6 7 6 9 2 9 6 9 7 5 6 5 7 9 6 6 2 2
 7 5 5 6 6 6 6 9 5 7 3 3 5 3 3 6 7 7 5 7 4 6 5 4 5 9 3 7 5 9 6 9 5 9 7 6 9
 3 7 2 5 7 5 6 6 5 9 2 5 7 7 9 5 7 5 7 2 5 7 6 5 3 2 6 6 6 3 6 6 3 5 3 7 7
 9 3 6 3 5 9 5 6 3 7 6 3 9 5 9 9 7 7 7 2 9 9 6 2 3 5 4 3 2 7 6 2 7 9 2 0 6
 9 3 7 9 0 2 5 5 7 9 2 5 5 7 2 4 3 5 9 1 9 6 2 5 9 7 1 6 3 6 7 7 7 9 7 5 5
 9 1 0 5 9 5 7 4 7 9 5 5 6 9 3 3 9 5 5 9 6 3 5 9 7 9 2 9 7 5 0 9 4 3 7 6 5
 9 5 9 5 7 3 7 9 0 7 5 5 7 3 2 9 4 5 0 6 0 6 0 4 2 6 9 5 9 6 9 3 9 5 6 6 9
 6 2 6 2 0 2 0 5 6 7 7 7 7 0 9 6 3 9 7 9 6 5 3 9 5 6 5 9 6 6 6 5 6 7 9 3 2
 6 5 9 9 9 6 9 6 7 6 7 3 9 7 6 7 6 6 7 9 9 6 3 5 1 9 2]
y_test(원래값) :  [2 6 2 7 2 2 9 7 2 9 0 3 7 7 9 2 4 3 0 1 5 1 4 2 6 9 2 6 2 4 2 6 0 4 6 6 8
 9 0 3 2 0 6 6 0 1 4 2 9 0 5 7 6 3 1 5 8 4 8 9 1 9 6 5 4 5 8 0 4 3 1 2 1 1
 4 0 5 6 3 8 6 5 5 7 1 6 9 6 1 3 7 4 6 4 4 6 5 4 0 6 3 7 5 9 6 3 5 5 7 3 9
 1 4 1 5 7 5 8 1 0 1 2 0 7 7 5 5 7 5 7 2 9 7 6 0 3 2 1 1 1 3 6 1 8 0 3 7 7
 1 8 1 8 0 3 5 1 3 7 6 6 9 0 9 3 4 4 7 1 8 9 8 2 3 0 4 8 2 7 1 2 7 3 2 0 6
 8 3 7 9 0 2 5 0 4 3 2 3 0 4 2 4 1 0 8 2 3 6 2 9 9 4 1 6 8 3 4 7 7 9 7 5 0
 5 2 0 5 1 5 7 4 7 9 9 0 6 5 8 9 3 3 0 5 6 8 9 5 4 9 2 9 4 9 0 5 4 8 4 8 0
 5 5 3 0 7 8 7 3 7 4 9 5 4 3 2 3 4 6 0 6 0 6 9 4 2 1 8 0 5 8 5 3 8 9 8 6 3
 6 1 8 2 0 2 7 5 6 7 4 7 4 0 8 1 3 9 7 1 1 9 1 9 0 2 5 3 1 1 8 0 6 4 9 8 2
 8 5 3 6 9 6 3 8 7 8 4 8 9 5 3 7 1 8 4 9 8 2 3 5 2 8 1]
accuracy_score :  0.46944444444444444
cnn
loss :  0.02694259211421013
accuracy :  0.9972222447395325
y_pred(예측값) :  [2 6 2 7 2 2 9 7 2 9 0 3 7 7 9 2 4 3 0 1 5 1 4 2 6 9 2 6 2 4 2 6 0 4 6 6 8
 9 0 3 2 0 6 6 0 1 4 2 9 0 5 7 6 3 1 5 8 4 8 9 1 9 6 5 4 5 8 0 4 3 1 2 1 1
 4 0 5 6 3 8 6 5 5 7 1 6 9 6 1 3 7 4 6 4 4 6 5 4 0 6 3 7 5 9 6 3 5 5 7 3 9
 1 4 1 5 7 5 8 1 0 1 2 0 7 7 5 5 7 5 7 2 9 7 6 0 3 2 1 1 1 3 6 1 8 0 3 7 7
 1 8 1 8 0 3 5 1 3 7 6 6 9 0 9 3 4 4 7 1 8 9 8 2 3 0 4 8 2 7 1 2 7 3 2 0 6
 8 3 7 9 0 2 5 0 4 3 2 3 0 4 2 4 1 0 8 2 3 6 2 9 9 4 1 6 8 3 4 7 7 9 7 5 0
 5 2 0 5 1 5 7 4 7 9 9 0 6 5 8 9 3 3 0 5 6 8 9 5 4 9 2 9 4 9 0 5 4 8 4 8 0
 5 5 3 0 7 8 7 3 7 4 9 5 4 3 2 3 4 0 0 6 0 6 9 4 2 1 8 0 5 8 5 3 8 9 8 6 3
 6 1 8 2 0 2 7 5 6 7 4 7 4 0 8 1 3 9 7 1 1 9 1 9 0 2 5 3 1 1 8 0 6 4 9 8 2
 8 5 3 6 9 6 3 8 7 8 4 8 9 5 3 7 1 8 4 9 8 2 3 5 2 8 1]
lstm
loss :  0.1309637427330017
accuracy :  0.9527778029441833
y_pred(예측값) :  [2 6 2 7 2 2 9 7 2 8 0 3 7 7 9 2 4 3 0 1 5 3 4 2 6 9 2 6 2 4 2 6 0 4 6 6 8
 9 0 3 2 0 6 6 0 1 4 2 9 0 5 7 6 3 3 5 8 4 8 9 1 9 6 5 4 5 8 0 4 3 1 2 1 1
 4 0 5 6 3 8 6 5 5 7 3 6 9 6 3 3 7 4 6 4 4 6 5 4 0 6 3 7 5 9 6 3 5 5 7 3 3
 1 4 1 5 7 5 8 1 0 1 2 0 7 7 5 5 7 5 7 2 8 7 6 0 9 2 1 1 1 3 6 1 8 0 3 7 7
 1 8 1 8 0 3 5 1 3 7 6 6 9 0 9 3 4 4 7 1 8 9 8 2 3 0 4 8 2 7 1 2 7 3 2 0 6
 8 3 7 9 0 7 5 0 4 3 2 3 0 4 2 4 1 0 8 2 3 6 2 9 9 4 1 6 8 3 4 7 7 9 7 5 0
 5 2 0 5 1 5 7 4 7 8 9 0 6 5 8 9 3 3 0 5 6 8 9 5 4 9 2 9 4 9 0 5 4 8 4 8 0
 5 5 3 0 8 8 7 3 7 4 9 5 4 3 2 3 4 6 0 6 0 6 3 4 2 1 8 0 5 8 5 3 8 9 8 6 3
 6 1 8 2 0 2 7 5 6 7 4 7 4 0 8 1 3 9 3 3 8 9 1 9 0 2 5 3 1 1 8 0 5 4 9 8 2
 8 5 3 6 9 6 3 8 7 8 4 8 9 5 3 7 1 8 4 9 8 2 9 5 2 8 1]

'''