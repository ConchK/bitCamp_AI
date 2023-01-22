from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model 
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
path= './_save/'

#1. 데이터
datasets= load_boston()
x=datasets.data
y=datasets.target
x_train, x_test, y_train, y_test= train_test_split(
        x,y, shuffle=True, random_state=333, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

#2. 모델구성(함수형)  


#3. 컴파일, 훈련


# model.save(path+ 'keras29_3_save_model.h5')
#  == model.save('./_save/keras29_3_save_model.h5')
#loss : 24.78493881225586


model= load_model(path+ 'keras29_3_save_model.h5')
#loss : 24.78493881225586

# >> 위치에 따라 모델과 가중치까지 저장할 수 있음.
# 최종 weight 까지 저장되기 때문에 바뀌지 않음.

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss :', loss)

'''
loss : 24.78493881225586
'''

