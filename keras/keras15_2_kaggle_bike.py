#hppts://www.kaggle.com/competitions/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/bike-sharing-demand/'
train_csv= pd.read_csv(path+ 'train.csv', index_col=0)  #0번째 컬럼이 데이터가 아니라 인덱스 임을 명시
# == train_csv= pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv= pd.read_csv(path+ 'test.csv', index_col=0)
submission=pd.read_csv(path+'sampleSubmission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)   #(10886, 11) ==x / .shape: 행과 열 크기

print(train_csv.columns)     #.columns: 속성을 이용하여 컬럼명을 모두 출력
#Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count'],
#      dtype='object')

print(train_csv.info())  # .info: 행과 열의 크기, 컬럼명, 컬럼을 구성하는 값의 자료형 등을 출력
print(test_csv.info())    
print(train_csv.describe())  #.describe: 요약 통계량

####결측치 처리 1. 제거 ####
print(train_csv.isnull().sum())  #.isnull: 결측값을 True(결측값 있음)/False(결측값 없음)로 나타냄 > .sum: 결측치의 수 표시
train_csv=train_csv.dropna()  #.dropna()/.dropna(axis=0) : 행 제거 /.dropna(axis=1) : 열 제거
print(train_csv.isnull().sum()) 
print(train_csv.shape)   #(1328, 10)
print(test_csv.shape)  #(6493, 8)

x=train_csv.drop(['casual', 'registered','count'], axis=1)
print(x)     #[10886 rows x 10 columns]
y= train_csv['count']
print(y)
print(y.shape) #(10886,)

x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )
print(x_train.shape, x_test.shape)   #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape)    #(7620,) (3266,)
print(submission.shape)  #(6493, 1)

#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=8, activation='linear'))   
#activation='linear(선형함수)' 디폴트
#linear한 연산의 레이어를 아무리 쌓아도 결국은 하나의 linear연산임.
#활성화 함수를 사용하면 선형분류기를 비선형 시스템으로 만들 수 있음. non-linear
model.add(Dense(30, activation='linear'))
model.add(Dense(70,activation='linear'))
model.add(Dense(100,activation='relu'))
model.add(Dense(150,activation='linear'))
model.add(Dense(100,activation='linear'))
model.add(Dense(70,activation='sigmoid'))
model.add(Dense(40,activation='linear'))
model.add(Dense(1, activation='linear'))   #만약 sigmiod 쓰면 값이 0~1, relu:0이하는 0으로 표시,히든 레이어에 사용

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start= time.time()
model.fit(x_train, y_train, epochs=300, batch_size=10)
end= time.time()

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)  #x_test,y_test 값으로 평가
print('loss :', loss)
y_predict=model.predict(x_test)  #x_test값으로 y_predict 예측
print('x_test :', x_test)
print('y_predict :', y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))
print('걸린시간: ', end-start)
'''
RMSE:  180.3301720099898
걸린시간:  975.5064024925232
'''

#제출
y_submit=model.predict(test_csv)
print(y_submit)
print(y_submit.shape)  #(715, 1)

#.to_csv()를 사용해서
#submission_0105.csv를 완성
print(submission)
submission['count']= y_submit
print(submission)

submission.to_csv(path+ 'sampleSubmission_01061233.csv')  #제출용 파일 생성

