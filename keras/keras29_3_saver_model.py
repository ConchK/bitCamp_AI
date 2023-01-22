from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model  
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
input1= Input(shape=(13,))
dense1= Dense(1)(input1)
dense2= Dense(15)(dense1)
dense3= Dense(5)(dense2)
output1= Dense(1)(dense3)
model= Model(inputs=input1, outputs=output1)   #정의가 마지막으로 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping    
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  
hist=model.fit(x_train, y_train, epochs=100, batch_size=1,
        validation_split=0.2, callbacks=[earlystopping],  #반환값
        verbose=1)
######################################################################

model.save(path+ 'keras29_3_save_model.h5')
#  == model.save('./_save/keras29_3_save_model.h5')

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss :', loss)

'''
loss : 24.78493881225586
'''

