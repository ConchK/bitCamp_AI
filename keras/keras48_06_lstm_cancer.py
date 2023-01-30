
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from sklearn.model_selection import train_test_split


#1. 데이터
datasets= load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)


x=datasets['data']
y=datasets['target']
# print(x.shape, y.shape)     #(569, 30) (569,)

x_train, x_test, y_train, y_test= train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123
        )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()

scaler.fit(x_train)
x_train= scaler.transform(x_train)
# x_train= scaler.fit_transform(x_train)    #x범위 만큼의 가중치 생성

x_test= scaler.transform(x_test)


print(x_train.shape, x_test.shape)      #(398, 30) (171, 30)

x_train= x_train.reshape(398,10,3)
x_test= x_test.reshape(171,10,3)

#2. 모델구성

model= Sequential()
model.add(LSTM(units = 40, input_shape= (10,3), activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation= 'relu'))  
model.add(Dense(40,activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', #이진분류는 무조건 loss='binary_crossentropy'/ BCE(x)=−1N∑i=1Nyilog(h(xi;θ))+(1−yi)log(1−h(xi;θ)) ??뭔 소린지..
                optimizer='adam', metrics=['accuracy'])  

import datetime
date= datetime.datetime.now()
date= date.strftime("%m%d_%H%M")   #str로 변형

path= './_save/MCP/'
name= 'k31_6_' +date +'_{epoch:04d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping= EarlyStopping(monitor='val_loss',  #'accuracy'들어가면 mode=max 지만, 'val_loss'가 더 적당함.
                             mode=min, patience=10, restore_best_weights=True, verbose=1)
# mcp= ModelCheckpoint(monitor= 'val_loss', mode=min, verbose=1, save_best_only=True , filepath=path +name)  #훈련결과만 저장

model.fit(x_train, y_train, epochs=150, batch_size=10, validation_split=0.2, callbacks=[earlystopping], verbose=2)

#4. 평가, 예측
# loss=model.evaluate(x_test,y_test)
# print('loss, accuracy: ', loss)
loss, accuracy= model.evaluate(x_test, y_test)
print('loss: ',loss)
print('accuracy: ',accuracy)



y_predict= model.predict(x_test)
# print(y_predict[:10])  
# print(y_test[:10])

y_predict=y_predict.astype('int')      #0~1로 나온 결과(y_predict)를 정수형 0 or 1로 바꾸면 accuracy_score에 쓸 수 있음

from sklearn.metrics import r2_score, accuracy_score   #accuracy_score( ) : 정답률/정확도. 실제 데이터 중 맞게 예측한 데이터의 비율
acc= accuracy_score(y_test, y_predict)
print('accuracy_score: ', acc)

'''
dnn
loss:  0.08399030566215515
accuracy:  0.9824561476707458
accuracy_score:  0.39766081871345027
cnn
loss:  0.07077265530824661
accuracy:  0.9766082167625427
accuracy_score:  0.39766081871345027
lstm
loss:  0.09619276970624924
accuracy:  0.9707602262496948
accuracy_score:  0.4269005847953216
'''