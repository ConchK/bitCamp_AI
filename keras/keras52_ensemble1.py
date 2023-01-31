import numpy as np

x1_datasets = np.array([range(100), range(301, 401)]).transpose()

#행과 열을 바꿔줘야 함.
# print(x1_datasets.shape)        #(2, 100)     >> .transpose()
print(x1_datasets.shape)        #(100, 2)  /삼성전자 시간, 고가

x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).T    #전치 
print(x2_datasets.shape)        #(100, 3)   / 아모레 시가, 고가, 종가

y = np.array(range(2001, 2101))   #(100,)   #삼성전자의 하루 뒤 종가

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test= train_test_split(
        x1_datasets, x2_datasets, y,
        shuffle=True, random_state=123
        )   

print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)
#  (75, 2) (25, 2) (75, 3) (25, 3) (75,) (25,)

#2. 모델구성(함수형)
from tensorflow. keras. models import Model  #함수형 모델 사용할 거라 Sequential은 import 안함.
from tensorflow.keras.layers import Dense, Input

#2-1.모델1
input1= Input(shape=(2,))
dense1= Dense(11, activation= 'relu', name= 'ds11')(input1)
dense2= Dense(12, activation= 'relu', name= 'ds12')(dense1)
dense3= Dense(13, activation= 'relu', name= 'ds13')(dense2)
output1= Dense(14, activation= 'relu', name= 'ds14')(dense3)

#2-2.모델2
input2= Input(shape=(3,))
dense21= Dense(21, activation= 'linear', name= 'ds21')(input2)
dense22= Dense(22, activation= 'linear', name= 'ds22')(dense21)
output2= Dense(23, activation= 'linear', name= 'ds23')(dense22)

#2-3.모델3
from tensorflow.keras.layers import concatenate     #모델1과 모델2를 엮어줌.
merge1= concatenate([output1, output2], name= 'mg1')
merge2= Dense(12, activation= 'relu', name= 'mg2')(merge1)
merge3= Dense(13, name= 'mg3')(merge2)
last_output= Dense(1, name= 'last')(merge3)
# y의 column이 1개라서 마지막 output이 1
model= Model(inputs=[input1, input2], outputs=last_output)   #정의가 마지막으로 
model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit([x1_train, x2_train], y_train, epochs=10, verbose=2, batch_size=10)

#4 평가
loss = model.evaluate([x1_test,x2_test], y_test)

print('loss: ',loss)







