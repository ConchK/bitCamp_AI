from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터
datasets= load_iris()
# print(datasets.DESCR)   # 판다스  .describe()  /  .info()
#상세내역 보기 x=3개 y=1개. class correlation에서 상관관계 확인. 쓸모없는 정보는 제외. 수치가 너무 높아도 안됨.
# print(datasets.feature_names)       #판다스   .columns

x= datasets.data   #이렇게도 작성 가능
y= datasets['target']
# print(x)
# print(y)
# print(x.shape, y.shape)    #(150, 4) (150,)

# ######원핫 인코딩######
# from tensorflow.keras.utils import to_categorical    
# y= to_categorical(y)
# print(y)
# print(y.shape)  #(150, 3)

x_train, x_test, y_train, y_test= train_test_split(
    x, y, shuffle=True,  #False의 문제점은 shuffle이 되지 않음. 
                        #True의 문제점은 특정 class를 제외할 수 없음.-데이터를 수집하다 보면 균형이 안맞는 경우.
                        #회귀는 데이터가 수치라서 상관 없음.
    stratify=y,     # y : yes  / 수치가 한 쪽으로 치우치는 걸 방지. y의 데이터가 분류형일 때만 가능.
    random_state=333,
    test_size=0.2
)
print(y_train)
print(y_test)

#2. 모델구성
model=Sequential()
model.add(Dense(5, activation='relu', input_shape=(4,)))   #or input_dim=4   
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(80,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(3,activation='softmax'))   #다중분류는 무조건 softmax. y_shape의 열x
                                            #y 종류(class)의 갯수. 다중분류(softmax)는 여러 개의 노드를 생성해서 100%에서 각각 나눔.  >  에러  > 원핫 인코딩 사용
                                            #One-Hot Encoding 데이터의 가치를 동일하게 만들기 위해 좌표 형태로 변환.
                                            #  | 0  1  2
                                            #-----------
                                            #0 | 1  0  0  =1
                                            #1 | 0  1  0  =1    > 단 하나의 값만 True이고 나머지는 모두 False
                                            #2 | 0  0  1  =1    >  y=(150, ) -원핫 인코딩-> (150,3) 
                                            #원핫은 안했지만 했을 거라 치고 y의 칼럼 갯수를 입력.

#3. 컴파일, 훈련         
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',    #원핫을 안했기 때문에 loss='sparse_categorical_crossentropy'
                metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=5, 
            validation_split=0.2, verbose=1)

#4. 평가, 예측
loss, accuracy= model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict= model.predict(x_test[:10])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict= model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)    #np.argmax :함수 내에 array와 비슷한 형태(리스트 등 포함)의 input을 넣으면 가장 큰 원소의 인덱스를 반환
                                           #가장 큰 원소가 여러개 있는 경우 가장 앞의 인덱스를 반환
print("y_pred(예측값) : ",y_predict)
# y_test= np.argmax(y_test, axis=1)  #원핫을 안했으니까 여기선 필요없음.
print("y_test(원래값) : ",y_test)
acc= accuracy_score(y_test, y_predict)
print(acc)


'''
loss :  0.09229588508605957
accuracy :  0.9333333373069763
y_pred(예측값) :  [2 0 2 2 1 1 2 0 2 0 0 0 0 2 2 2 0 2 0 1 2 0 1 1 2 0 1 1 1 2]
y_test(원래값) :  [2 0 2 1 1 1 2 0 2 0 0 0 0 2 2 2 0 2 0 1 2 0 1 1 2 0 1 1 1 1]
0.9333333333333333
'''