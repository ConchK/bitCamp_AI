
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

#2. 모델구성
model=Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))   
model.add(Dense(140,activation='sigmoid'))
model.add(Dense(100,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(1,activation='sigmoid'))  #이진분류는 최종 레이어 activation='sigmoid'(0~1)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', #이진분류는 무조건 loss='binary_crossentropy'/ BCE(x)=−1N∑i=1Nyilog(h(xi;θ))+(1−yi)log(1−h(xi;θ)) ??뭔 소린지..
                optimizer='adam', metrics=['accuracy'])  

from tensorflow.keras.callbacks import EarlyStopping
earlystopping= EarlyStopping(monitor='val_loss',  #'accuracy'들어가면 mode=max 지만, 'val_loss'가 더 적당함.
                             mode=min, patience=20, restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2, callbacks=[earlystopping], verbose=1)

#4. 평가, 예측
# loss=model.evaluate(x_test,y_test)
# print('loss, accuracy: ', loss)
loss, accuracy= model.evaluate(x_test, y_test)
print('loss: ',loss)
print('accuracy: ',accuracy)



y_predict= model.predict(x_test)
print(y_predict[:10])  
print(y_test[:10])

y_predict=y_predict.astype('int')      #0~1로 나온 결과(y_predict)를 정수형 0 or 1로 바꾸면 accuracy_score에 쓸 수 있음

from sklearn.metrics import r2_score, accuracy_score   #accuracy_score( ) : 정답률/정확도. 실제 데이터 중 맞게 예측한 데이터의 비율
acc= accuracy_score(y_test, y_predict)
print('accuracy_score: ', acc)





'''
loss:  0.0893794521689415
accuracy:  0.9707602262496948
accuracy_score:  0.39766081871345027

'''