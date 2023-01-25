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
# print(x.shape, y.shape)   #(506, 13) (506,)

x_train, x_test, y_train, y_test= train_test_split(
        x,y, shuffle=True, random_state=333, test_size=0.2
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()

scaler.fit(x_train)
x_train= scaler.transform(x_train)
# == x_train= scaler.fit_transform(x_train)   
x_test= scaler.transform(x_test)

#2. 모델구성(함수형)  
input1= Input(shape=(13,))
dense1= Dense(1)(input1)
dense2= Dense(15)(dense1)
dense3= Dense(5)(dense2)
output1= Dense(1)(dense3)
model= Model(inputs=input1, outputs=output1)   #정의가 마지막으로 
model.summary()
# Total params: 130

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=0)  

mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +'MCP/keras30_ModelCheckPoint1.hdf5')

hist=model.fit(x_train, y_train, epochs=100, batch_size=1,
        validation_split=0.2, callbacks=[earlystopping, mcp],  #반환값, 2개 이상은 리스트로
        verbose=0)
#hist는 dictionary 형태로(key value) loss값이 들어있음/ 리스트 2개 이상
#hist를 통해 CheckPoint를 확인하고 가중치 저장.

# Epoch 00083: val_loss did not improve from 16.12872  > 저장 안됨.

# Epoch 00084: val_loss improved from 16.12872 to 16.01115, saving model to ./_save/MCP\keras30_ModelCheckPoint1.hdf5 
# > 갱신 될 때 마다 가장 좋은 저장치가 파일에 저장 됨.
#loss : 22.809310913085938

# model.save(path+ 'MCP/keras30_ModelCheckPoint1.hdf5')  <--없어도 됨.


#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss :', loss)

'''
loss : 22.725204467773438
'''

