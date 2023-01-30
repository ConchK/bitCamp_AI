from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import fetch_covtype        #####print(datasets.get_data_home()) import한 datasets 들어있는 곳
from sklearn.model_selection import train_test_split
import pandas as pd 

#1. 데이타
datasets= fetch_covtype()
x= datasets.data
y= datasets.target
print(x.shape, y.shape)   #(581012, 54) (581012,)
print(np.unique(y, return_counts=True))   #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))

'''
########### 1.keras to_categorical
from tensorflow.keras.utils import to_categorical    #0번째 .delete
y= to_categorical(y)    #원핫인코딩
print(y.shape)   #(581012, 8) 앞 0자리를 0으로 채움
print(type(y))   #<class 'numpy.ndarray'>
print(y[:10])
# [[0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]]
print(np.unique(y[:,0], return_counts=True))   #(array([0.], dtype=float32), array([581012], dtype=int64)) >0만 581012개 있음.
print(np.unique(y[:,1], return_counts=True))   #(array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))
print("==================================")
y= np.delete(y, 0, axis=1)   #(581012, 7)
print(y.shape)
# [[0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]]
print(y[:10])
print(np.unique(y[:,0], return_counts=True))   #(array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))

'''
'''
############ 2. pandas get_dummies
import pandas as pd       #자료형 확인  print(type(y))   .values   .numpy() mp.ros
# print(datasets.feature_names)
y = pd.get_dummies(y)   # >get_dummies 후 판다스 형태가 됨(헤더와 인덱스가 있음.)  >np자료형이 판다스를 못 받아들임. numpy로 바꿔야 함.
print(y[:10])
#    1  2  3  4  5  6  7   ->헤더
# 0  0  0  0  0  1  0  0
# 1  0  0  0  0  1  0  0
# 2  0  1  0  0  0  0  0
# 3  0  1  0  0  0  0  0
# 4  0  0  0  0  1  0  0
# 5  0  1  0  0  0  0  0
# 6  0  0  0  0  1  0  0
# 7  0  0  0  0  1  0  0
# 8  0  0  0  0  1  0  0
# 9  0  0  0  0  1  0  0
# >인덱스
print(type(y))   #<class 'pandas.core.frame.DataFrame'> 형태. 인덱스와 헤더(컬럼명)가 자동 생성.
print(y.shape)   #(581012, 7)
# > y_test를 numpy로 바꿔야 함. 2가지 방법.
y= y.values
# y= y.to_numpy()  
print(type(y))
'''


############### 3. onehotencoder
from sklearn.preprocessing import OneHotEncoder      #preprocessing :전처리  
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()    #LabelEncoder로 숫자로 변환      
# encoder.fit(y)
# labels = encoder.transform(y)

# labels = labels.reshape(-1,1)       #2차원 데이터로 변환

print(y.shape)   #(581012,)
y= y.reshape(581012,1)  #1D 벡터 형태를 2D로 shape를 바꿈.
ohe =OneHotEncoder() 
print(y.shape)
y= ohe.fit_transform(y)   # ohe.fit(y) / y= ohe.transform(y)
y= y.toarray()
print(y[:15])
print(type(y))
print(y.shape)

# oh_encoder.fit(labels)
# oh_labels = oh_encoder.transform(labels)

# print('원-핫 인코딩 데이터')
# print(oh_labels.toarray())
# print('원-핫 인코딩 데이터 차원')
# print(oh_labels.shape)   #(581012, 7)
####################################


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
model.add(Dense(5, activation='relu', input_shape=(54,)))   #or input_dim=54   
model.add(Dense(90,activation='sigmoid'))
model.add(Dense(150,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(7,activation='softmax'))  

#3. 컴파일, 훈련         
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping   
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  


model.fit(x_train, y_train, epochs=2, batch_size=256, 
            validation_split=0.2, verbose=1, callbacks=[earlystopping])

#4. 평가, 예측
loss, accuracy= model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict= model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)    #가장 큰 위치값을 찾아냄
print("y_pred(예측값) : ",y_predict)

y_test= np.argmax(y_test, axis=1)
print("y_test(원래값) : ",y_test)  

acc= accuracy_score(y_test, y_predict)
print("accuracy_score : ",acc)

'''
loss :  0.7558484077453613
accuracy :  0.6800254583358765
y_pred(예측값) :  [1 0 1 ... 1 1 0]
y_test(원래값) :  [1 6 4 ... 1 1 0]
accuracy_score :  0.680025472664217
'''