import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path= './_data/ddarung/'   
train_csv= pd.read_csv(path+ 'train.csv', index_col=0)  #0번째 컬럼이 데이터가 아니라 인덱스 임을 명시 / x값
# train_csv= pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv= pd.read_csv(path+ 'test.csv', index_col=0)
submission=pd.read_csv(path+'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)   #(1459, 10)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

print(train_csv.info())    #결측치 처리- 결측치가 있는 데이터를 삭제
print(test_csv.info())    
print(train_csv.describe())

x= train_csv.drop(['count'], axis=1)   #train_csv에서 count를 제외
print(x)     #[1459 rows x 9 columns]
y= train_csv['count']
print(y)
print(y.shape)  #(1459,)

x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )
print(x_train.shape, x_test.shape)    #(1021, 9) (438, 9)
print(y_train.shape, y_test.shape)    #(1021,) (438,)

#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(140))
model.add(Dense(160))
model.add(Dense(180))
model.add(Dense(200))
model.add(Dense(220))
model.add(Dense(250))
model.add(Dense(170))
model.add(Dense(150))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
loss=model.evaluate(x_test,y_test)  #x_test,y_test 값으로 평가
print('loss :', loss)
y_predict=model.predict(x_test)  #x_test값으로 y_predict 예측
print('x_test :', x_test)
print('y_predict :', y_predict)

######결측치 > nan > 실패########

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

#제출
y_submit=model.predict(test_csv)



