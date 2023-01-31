#https://www.kaggle.com/competitions/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/bike-sharing-demand/'
train_csv= pd.read_csv(path+ 'train.csv', index_col=0)  #0번째 컬럼이 데이터가 아니라 인덱스 임을 명시
# == train_csv= pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv= pd.read_csv(path+ 'test.csv', index_col=0)
submission=pd.read_csv(path+'sampleSubmission.csv', index_col=0)

# print(train_csv)
# print(train_csv.shape)   #(10886, 11) ==x / .shape: 행과 열 크기

# print(train_csv.columns)     #.columns: 속성을 이용하여 컬럼명을 모두 출력
# #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
# #       'humidity', 'windspeed', 'casual', 'registered', 'count'],
# #      dtype='object')

# print(train_csv.info())  # .info: 행과 열의 크기, 컬럼명, 컬럼을 구성하는 값의 자료형 등을 출력
# print(test_csv.info())    
# print(train_csv.describe())  #.describe: 요약 통계량

####결측치 처리 1. 제거 ####
# print(train_csv.isnull().sum())  #.isnull: 결측값을 True(결측값 있음)/False(결측값 없음)로 나타냄 > .sum: 결측치의 수 표시
train_csv=train_csv.dropna()  #.dropna()/.dropna(axis=0) : 행 제거 /.dropna(axis=1) : 열 제거
# print(train_csv.isnull().sum()) 
# print(train_csv.shape)   #(1328, 10)
# print(test_csv.shape)  #(6493, 8)

x=train_csv.drop(['casual', 'registered','count'], axis=1)
# print(x)     #[10886 rows x 10 columns]
y= train_csv['count']
# print(y)
# print(y.shape) #(10886,)




x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )
# print(x_train.shape, x_test.shape)   #(7620, 8) (3266, 8)
# print(y_train.shape, y_test.shape)    #(7620,) (3266,)
# print(submission.shape)  #(6493, 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()

scaler.fit(x_train)
x_train= scaler.transform(x_train)
# x_train= scaler.fit_transform(x_train)    #x범위 만큼의 가중치 생성, x의 train만

x_test= scaler.transform(x_test)

test_csv= scaler.transform(test_csv)

print(x_train.shape, x_test.shape, test_csv.shape)      #(7620, 8) (3266, 8) (6493, 8)

x_train= x_train.reshape(7620,8,1)
x_test= x_test.reshape(3266,8,1)
test_csv= test_csv.reshape(6493,8,1)

# #2. 모델구성

model= Sequential()
model.add(Conv1D(40, 2, input_shape= (8,1), activation= 'relu'))
model.add(Conv1D(30, 2, activation= 'relu'))
model.add(Conv1D(20, 2, activation= 'relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(80,activation= 'relu'))  
model.add(Dense(40,activation= 'relu'))
model.add(Dense(1, activation= 'linear'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k31_5_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss', mode=min, patience=10, restore_best_weights=True, verbose=1)

# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name)  #훈련결과만 저장

hist=model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2, 
                callbacks=[earlystopping], verbose=2)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)  #x_test,y_test 값으로 평가
print('loss :', loss)
y_predict=model.predict(x_test)  #x_test값으로 y_predict 예측
# print('x_test :', x_test)
# print('y_predict :', y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

'''
Conv2D
loss : [21974.208984375, 108.91911315917969]
RMSE:  148.23700232916593
Conv1D
loss : [22565.0, 111.15676879882812]
RMSE:  150.2165147055315

'''

#제출
y_submit=model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)  #(715, 1)

#.to_csv()를 사용해서
#submission_0105.csv를 완성
# print(submission)
submission['count']= y_submit
# print(submission)

# submission.to_csv(path+ 'sampleSubmission_01112305.csv')  #제출용 파일 생성

# print("=====================================")
# print(hist) 
# print(hist.history)
# print("=====================================")

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')   #리스트 형태는 x를 명시 하지 않아도 됨. 어차피 앞에서 부터.
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')  
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('boston loss')
# plt.legend(loc='upper left')  #location지정하지 않으면 그래프가 없는 지점에 자동으로 생성

# plt.show()