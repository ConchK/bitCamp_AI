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
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=False, verbose=1)  

mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +'MCP/keras30_ModelCheckPoint3.hdf5')  #훈련결과만 저장

hist=model.fit(x_train, y_train, epochs=100, batch_size=1,
        validation_split=0.2, callbacks=[earlystopping, mcp],  #반환값, 2개 이상은 리스트로
        verbose=1)
#hist는 dictionary 형태로(key value) loss값이 들어있음/ 리스트 2개 이상
#hist를 통해 CheckPoint를 확인하고 가중치 저장.

# Epoch 00083: val_loss did not improve from 16.12872  > 저장 안됨.

# Epoch 00084: val_loss improved from 16.12872 to 16.01115, saving model to ./_save/MCP\keras30_ModelCheckPoint1.hdf5 
# > 갱신 될 때 마다 가장 좋은 저장치가 파일에 저장 됨.
#loss : 22.809310913085938

model.save(path +'keras30_ModelCheckPoint3_save_model.h5')   #2.모델 + 3.컴파일, 훈련의 가중치 저장

# model= load_model(path +'MCP/keras30_ModelCheckPoint1.hdf5')
#데이터는 모델에 맞춰줘야 함.


#4. 평가, 예측
print("================1. 기본 출력==========================") 
loss=model.evaluate(x_test, y_test)
print('loss :', loss)

print("================2. load model 출력==========================") 
model2= load_model(path +'keras30_ModelCheckPoint3_save_model.h5')
loss=model2.evaluate(x_test, y_test)
print('loss :', loss)

print("================3. ModelCheckPoint출력==========================") 
model3= load_model(path +'MCP/keras30_ModelCheckPoint3.hdf5')
loss=model3.evaluate(x_test, y_test)
print('loss :', loss)

'''
restore_best_weights=True 했을 때,

================1. 기본 출력==========================   
4/4 [==============================] - 0s 1ms/step - loss: 22.9249
loss : 22.924915313720703
================2. load model 출력==========================
4/4 [==============================] - 0s 0s/step - loss: 22.9249
loss : 22.924915313720703
================3. ModelCheckPoint출력==========================     patience=20 이후에 가장 좋았던 loss의 결과를 저장 
4/4 [==============================] - 0s 2ms/step - loss: 22.9249
loss : 22.924915313720703

restore_best_weights=False 했을 때,    

================1. 기본 출력==========================
4/4 [==============================] - 0s 2ms/step - loss: 22.9234
loss : 22.923423767089844
================2. load model 출력==========================
4/4 [==============================] - 0s 5ms/step - loss: 22.9234
loss : 22.923423767089844
================3. ModelCheckPoint출력==========================    patience=20 이후의 결과를 저장
4/4 [==============================] - 0s 0s/step - loss: 23.7083
loss : 23.708314895629883

>>      val_loss :validation 으로 평가.
        loss :train data 로 평가.
        평가, 예측 evaluate :test data 사용
        Early stopping이 validation을 기준으로 평가한 결과, val_loss가 가장 객관적.
'''