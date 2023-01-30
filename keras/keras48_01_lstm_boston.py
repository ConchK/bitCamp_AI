#31_1

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
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

print(x_train.shape, x_test.shape)   # #(404, 13) (102, 13)  내용물, 순서가 바뀌지 않도록 reshape >> (13,1,1)=input_shape

x_train= x_train.reshape(404,13,1)
x_test= x_test.reshape(102,13,1)



#2. 모델구성(순차형)
model= Sequential()
# model.add(Dense(5, input_dim=13))   #(506,13)>(13)   kernel_size를 직사각형 형태로 바꿔줌. 
model.add(LSTM(units = 40, input_shape= (13,1), activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation= 'relu'))  
model.add(Dense(60,activation= 'relu'))
model.add(Dense(40,activation= 'relu'))
model.add(Dense(1, activation= 'linear'))
model.summary()
# Total params: 130 

#3. 컴파일, 훈련   
model.compile(loss='mse', optimizer='adam')    

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)  

hist=model.fit(x_train, y_train, epochs=100, batch_size=1,                                 
        validation_split=0.2, callbacks=[earlystopping], 
        verbose=2)



#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss :', loss)

from sklearn.metrics import mean_squared_error,r2_score
y_predict=model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   
print('RMSE :', RMSE(y_test, y_predict))

r2=r2_score(y_test, y_predict)    
print("R2 :", r2)

'''
cnn
oss : 584.281494140625
RMSE : 24.17191491355692
R2 : -4.957255332962484
lstm
loss : 24.680702209472656
RMSE : 4.9679678221649395
R2 : 0.7483588568064892

'''