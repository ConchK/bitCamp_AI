import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path= './_data/ddarung/'   #pd.read_csv(): pandas로 엑셀파일csv 불러오기
train_csv= pd.read_csv(path+ 'train.csv', index_col=0)  #0번째 컬럼이 데이터가 아니라 인덱스 임을 명시
# == train_csv= pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv= pd.read_csv(path+ 'test.csv', index_col=0)
submission=pd.read_csv(path+'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)   #(1459, 10) ==x / .shape: 행과 열 크기

print(train_csv.columns)     #.columns: 속성을 이용하여 컬럼명을 모두 출력
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

print(train_csv.info())  # .info: 행과 열의 크기, 컬럼명, 컬럼을 구성하는 값의 자료형 등을 출력
print(test_csv.info())    
print(train_csv.describe())  #.describe: 요약 통계량

####결측치 처리 1. 제거 ####
print(train_csv.isnull().sum())  #.isnull: 결측값을 True(결측값 있음)/False(결측값 없음)로 나타냄 > .sum: 결측치의 수 표시
train_csv=train_csv.dropna()  #.dropna()/.dropna(axis=0) : 행 제거 /.dropna(axis=1) : 열 제거
print(train_csv.isnull().sum()) 
print(train_csv.shape)   #(1328, 10)

x= train_csv.drop(['count'], axis=1)   #train_csv에서 count를 제외
print(x)     #[1459 rows x 9 columns]
y= train_csv['count']
print(y)
print(y.shape)  #(1459,)

x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )
print(x_train.shape, x_test.shape)    #(929, 9) (399, 9)
print(y_train.shape, y_test.shape)    #(929,) (399,)
print(submission.shape)  #(715, 1)

#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(40))
model.add(Dense(120))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start= time.time()
model.fit(x_train, y_train, epochs=100, batch_size=10)
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

'''
RMSE:  55.046864318003436
cpu걸린시간:  21.603867530822754
gpu걸린시간:  20.912834405899048
'''
print('걸린시간: ', end-start)


#제출
y_submit=model.predict(test_csv)
print(y_submit)
print(y_submit.shape)  #(715, 1)

#.to_csv()를 사용해서
#submission_0105.csv를 완성
print(submission)
submission['count']= y_submit
print(submission)

submission.to_csv(path+ 'submission_01050531.csv')  #제출용 파일 생성

