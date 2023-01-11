import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([range(10), range(21, 31), range(201, 211)])  #0부터 10개의 수=10개의 데이타 / 21~30 / 201~210 ,마지막 수-1
# print(range(10))
print(x.shape)  #(3, 10)
y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])

x=x.T
y=y.T
print(x.shape, y.shape)   #(10, 3) (10, 2)

#[실습] train_test_split를 이용하여 7:3

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


model=Sequential()
model.add(Dense(5,input_dim=3))  #x열의 갯수=input_dim
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(45))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(2))  #y열의 갯수

model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=200, batch_size=3)

loss=model.evaluate(x,y)
print('loss= ',loss)

result=model.predict([[9,30,210]])
print('9,30,210의 예측값 :',result)

'''
결과 :
loss=  0.8741191029548645
9,30,210의 예측값 : [[11.476744   1.1946892]]
'''
