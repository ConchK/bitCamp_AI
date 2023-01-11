import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터  -train set / test set 분리 
x=np.array([1,2,3,4,5,6,7,8,9,10])  #(10, ) 위치: 1=0번째, 10=9번째
y=np.array(range(10))               #(10, )

#실습 : 넘파이 리스트 슬라이싱 7:3
# x_train=x[:-3]  #=[:7]
# x_test=x[-3:]   #=[7:]
# y_train=y[:-3]  #=[:7]
# y_test=y[-3:]   #=[7:]

#[검색] train과 test를 섞어서 7:3으로 (힌트. 사이킷런)
# returns X_train, X_test, y_train, y_test dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.7, # train/test size 하나만 명시해도 됨
    test_size=0.3,  #test 30%=0.3
    shuffle=True,  #shuffle=True 무작위 추출, False 순차적 추출 /디폴트는 True
    random_state=123  #재현가능성을 위한 난수 초기값으로 아무 숫자나 지정/테스트 할 땐 동일하게 유지해야 같은 결과
    )

print('x_train :', x_train)   #추출 된 x,y 위치 동일
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. 모델구성 
model=Sequential()
model.add(Dense(8, input_dim=1))
model.add(Dense(13))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1)

#4.평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss :', loss)
result=model.predict([11])
print('[11의 결과 :', result)

'''
결과 :
loss : 0.045782219618558884
[11의 결과 : [[10.05628]]
'''