from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets= load_boston()
x=datasets.data
y=datasets.target
print(x.shape, y.shape)   #(506, 13) (506,)

x_train, x_test, y_train, y_test= train_test_split(
        x,y, shuffle=True, random_state=333, test_size=0.2
)
import time
start=time.time()

#2. 모델구성
model= Sequential()
# model.add(Dense(5, input_dim=13))   #(506,13)>(13)
model.add(Dense(200, input_shape=(13,)))   
#다차원은 input_dim 말고 input_shape로 표현/ ex.스칼라 (100,10,5)>(10,5), (506,13)>(13,)
#input_shape에서 (1,)은 행의 갯수를 뜻하고 input_dim에서 1은 입력 차원을 뜻한다.
model.add(Dense(150))
model.add(Dense(130))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train, y_train, epochs=20, batch_size=1,
        validation_split=0.2,
        verbose=1)
#hist는 dictionary 형태로(key value) loss값이 들어있음/ 리스트 2개 이상
end=time.time()

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
# print('loss :', loss)

# print('걸린시간: ',end-start )

# print("=====================================")
# print(hist)  #<keras.callbacks.History object at 0x00000257928543D0>
# print(hist.history)
# print("=====================================")
# print(hist.history['loss'])
# print("=====================================")
# print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   #'맑은 고딕' 폰트 사용 가능

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')   #리스트 형태는 x를 명시 하지 않아도 됨. 어차피 앞에서 부터.
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')  
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('보스턴 손실함수')  
plt.legend(loc='upper left')  #location지정하지 않으면 그래프가 없는 지점에 자동으로 생성

plt.show()


