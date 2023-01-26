from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
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

######원핫 인코딩######
from tensorflow.keras.utils import to_categorical    
y= to_categorical(y)
print(y)
print(y.shape)  #(150, 3)

x_train, x_test, y_train, y_test= train_test_split(
    x, y, shuffle=True,  #False의 문제점은 shuffle이 되지 않음. 
    stratify=y,     # y : yes  / 수치가 한 쪽으로 치우치는 걸 방지. y의 데이터가 분류형일 때만 가능.
    random_state=333,
    test_size=0.2
)
# print(y_train)
# print(y_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()

scaler.fit(x_train)
x_train= scaler.transform(x_train)
# x_train= scaler.fit_transform(x_train)    #x범위 만큼의 가중치 생성
x_test= scaler.transform(x_test)

print(x_train.shape, x_test.shape)      #(120, 4) (30, 4)

x_train= x_train.reshape(120,2,2,1)
x_test= x_test.reshape(30,2,2,1)


# #2. 모델구성
# model=Sequential()
# model.add(Dense(20, activation='relu', input_shape=(4,)))   #or input_dim=4   
# model.add(Dense(80,activation='sigmoid'))
# model.add(Dropout(0.5))
# model.add(Dense(160,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(60,activation='relu'))
# model.add(Dense(40,activation='linear'))
# model.add(Dense(3,activation='softmax'))   #다중분류는 무조건 softmax. y_shape의 열x
# model.summary()    
# #Total params: 39,463         

model= Sequential()
model.add(Conv2D(40, (2,1), input_shape= (2,2,1), padding = 'same', activation= 'relu'))
model.add(Conv2D(40, (2,1), padding = 'same', activation= 'relu'))
model.add(Flatten())
model.add(Dense(120, activation= 'relu'))  
model.add(Dense(60,activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))
model.summary()      

#3. 컴파일, 훈련         
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k31_7_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss',  #'accuracy'들어가면 mode=max 지만, 'val_loss'가 더 적당함.
                             mode=min, patience=20, restore_best_weights=True, verbose=1)
# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name)  #훈련결과만 저장

model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[earlystopping], verbose=2)

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
y_test= np.argmax(y_test, axis=1)
print("y_test(원래값) : ",y_test)
acc= accuracy_score(y_test, y_predict)
print(acc)

'''
dnn
loss :  0.17746999859809875
accuracy :  0.9333333373069763
y_pred(예측값) :  [0 2 0 2 2 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
y_test(원래값) :  [0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
0.9333333333333333
cnn
loss :  0.18217000365257263
accuracy :  0.8999999761581421
y_pred(예측값) :  [0 2 0 2 2 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
y_test(원래값) :  [0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
0.9

'''