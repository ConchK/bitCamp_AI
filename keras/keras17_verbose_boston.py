from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets= load_boston()
x=datasets.data
y=datasets.target
print(x.shape, y.shape)   #(506, 13) (506,) >>

x_train, x_test, y_train, y_test= train_test_split(
        x,y, shuffle=True, random_state=333, test_size=0.2
)
import time
start=time.time()

print('shape')
print(x_train.shape, x_train.ndim)

#2. 모델구성
model= Sequential()
# model.add(Dense(5, input_dim=13))   #(506,13)>(13)
model.add(Dense(5, input_shape=(13,)))   
#다차원은 input_dim 말고 input_shape로 표현/ ex.스칼라 (100,10,5)>(10,5), (506,13)>(13,)
#input_shape에서 (1,)은 행의 갯수를 뜻하고 input_dim에서 1은 입력 차원을 뜻한다.
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=1,
        validation_split=0.2,
        verbose=2)  
# 0 = silent, 1 = progress bar, 2 = one line per epoch.
# 결과만 보여주기 때문에 0이 제일 빠름

end=time.time()

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss :', loss)

print('걸린시간: ',end-start )

'''
loss : 73.31171417236328
걸린시간:  8.836705684661865
'''