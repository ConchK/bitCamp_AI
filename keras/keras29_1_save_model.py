#시퀀셜 탈피
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model  #인풋 레이어를 명시해야 함.
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np


#1. 데이터
datasets= load_boston()
x=datasets.data
y=datasets.target
# print(x.shape, y.shape)   #(506, 13) (506,)

x_train, x_test, y_train, y_test= train_test_split(
        x,y, shuffle=True, random_state=333, test_size=0.2
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()

scaler.fit(x_train)
x_train= scaler.transform(x_train)
# x_train= scaler.fit_transform(x_train)    #x범위 만큼의 가중치 생성

x_test= scaler.transform(x_test)

# print(x)
# print(type(x))   #<class 'numpy.ndarray'>
# print("최소값 : ", np.min(x))
# print("최대값 : ", np.max(x))

#2. 모델구성(함수형)  
input1= Input(shape=(13,))
dense1= Dense(1)(input1)
dense2= Dense(15)(dense1)
dense3= Dense(5)(dense2)
output1= Dense(1)(dense3)
model= Model(inputs=input1, outputs=output1)   #정의가 마지막으로 
model.summary()
# Total params: 130

path= './_save/'
model.save(path+ 'keras29_1_save_model.h5')
#  == model.save('./_save/keras29_1_save_model.h5')


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping    
# earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  
# #history의 val_loss 값 사용/ loss는 최소값을 갱신해야 함으로 min사용. /accuracy는 높으면 좋기 때문에 max./ 모르겠으면 outo/ patience :갱신 안되는 걸 10번 참겟다..
# #최적의 weight 후 10번에서 stop > restore_best_weights=True 사용해서 해결.
# hist=model.fit(x_train, y_train, epochs=100, batch_size=1,
#         validation_split=0.2, callbacks=[earlystopping],  #반환값
#         verbose=1)
# #hist는 dictionary 형태로(key value) loss값이 들어있음/ 리스트 2개 이상

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss :', loss)

'''
loss : 24.28295135498047
'''

