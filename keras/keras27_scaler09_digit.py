from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

#2. 모델구성
model=Sequential()
model.add(Dense(5, activation='relu', input_shape=(64,)))   #or input_dim=64   
model.add(Dense(90,activation='sigmoid'))
model.add(Dense(150,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(10,activation='softmax'))  

#3. 컴파일, 훈련         
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping   
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  


model.fit(x_train, y_train, epochs=300, batch_size=5, 
            validation_split=0.2, verbose=1, callbacks=[earlystopping])

#4. 평가, 예측
loss, accuracy= model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict= model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)    #가장 큰 위치값을 찾아냄
print("y_pred(예측값) : ",y_predict)

y_test= np.argmax(y_test, axis=1)
print("y_test(원래값) : ",y_test)  

acc= accuracy_score(y_test, y_predict)
print("accuracy_score : ",acc)

'''
전 :
loss :  0.19437795877456665
accuracy :  0.949999988079071
y_pred(예측값) :  [2 6 2 7 2 2 9 7 2 4 0 3 7 7 9 2 4 2 0 1 5 1 4 2 6 0 2 6 2 4 2 6 0 4 6 6 8
 9 0 3 2 0 6 6 0 1 4 2 9 0 5 7 6 3 1 5 1 4 8 9 1 9 6 5 4 5 8 0 4 3 1 2 1 1
 4 0 5 6 3 8 6 5 5 7 1 6 9 6 1 3 8 4 6 4 4 6 5 4 0 6 3 7 5 9 6 3 5 5 7 3 8
 1 4 1 5 7 5 8 1 0 1 2 0 7 7 5 5 7 5 7 2 9 7 6 0 3 2 1 1 1 3 6 1 8 0 3 7 7
 1 8 1 8 0 3 5 1 3 7 6 6 9 0 9 3 7 4 7 1 8 9 8 2 3 0 4 8 2 7 1 2 7 3 2 0 6
 8 3 7 9 0 1 5 0 4 3 2 3 0 4 2 4 1 0 8 2 3 6 2 9 9 4 1 6 7 3 4 7 7 9 7 5 0
 5 2 0 5 9 5 7 4 7 8 9 0 6 5 8 9 3 3 0 5 6 8 9 5 4 9 2 9 4 9 0 5 4 8 4 7 5
 5 5 3 5 7 8 7 3 2 4 9 5 4 3 2 3 4 0 0 6 0 6 9 4 2 1 8 0 5 7 5 3 8 9 8 6 3
 6 1 8 2 0 2 7 5 6 7 4 7 4 0 8 1 3 9 7 1 1 9 1 9 0 2 5 3 1 1 8 0 1 4 9 8 2
 8 5 3 6 9 6 3 8 7 8 4 8 9 5 3 7 1 8 4 9 8 2 3 5 2 8 1]
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
accuracy_score :  0.95
후 :
loss :  0.3035730719566345
accuracy :  0.8916666507720947
y_pred(예측값) :  [2 6 2 7 2 9 9 7 8 9 0 3 7 7 9 1 4 3 0 1 5 8 4 2 6 9 2 6 3 1 1 6 0 4 6 6 8
 9 0 3 2 0 6 6 0 4 4 2 9 0 5 7 6 3 1 5 8 4 8 9 1 9 4 5 4 5 8 0 4 3 8 4 8 1
 4 0 5 6 2 8 6 5 5 7 1 6 9 6 8 2 7 4 6 4 4 6 5 4 0 6 3 7 5 9 6 3 5 5 7 3 3
 8 4 1 5 7 5 8 1 0 1 2 0 7 7 5 5 7 5 7 2 9 7 6 0 3 2 1 1 1 3 6 1 8 0 3 7 7
 1 8 1 8 0 3 5 1 3 7 6 6 9 0 9 3 4 4 7 1 8 9 8 2 3 0 4 8 3 7 1 2 7 3 2 0 6
 8 3 7 9 0 2 5 0 4 3 2 3 0 4 2 4 1 0 5 2 3 6 2 9 9 4 8 6 6 3 4 7 9 9 7 5 0
 5 2 0 5 8 5 7 4 7 3 0 0 6 5 8 9 3 3 0 5 6 8 9 5 4 9 2 9 4 9 0 5 4 8 4 4 0
 5 5 3 0 7 8 7 3 9 4 9 5 4 3 2 3 4 0 0 6 0 6 9 4 2 1 8 0 5 8 5 3 5 9 8 6 3
 6 1 8 2 0 2 9 5 6 7 4 4 4 0 8 1 3 9 7 8 1 9 8 9 0 3 5 3 1 1 8 0 6 4 9 8 2
 8 5 3 6 9 6 3 8 7 3 4 8 9 5 3 7 8 8 4 9 8 3 8 5 8 8 1]
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
accuracy_score :  0.8916666666666667

'''