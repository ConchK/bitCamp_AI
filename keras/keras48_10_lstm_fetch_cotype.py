from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
import numpy as np
from sklearn.datasets import fetch_covtype        #####print(datasets.get_data_home()) import한 datasets 들어있는 곳
import pandas as pd 
from sklearn.model_selection import train_test_split


#1. 데이타
datasets= fetch_covtype()
x= datasets.data
y= datasets.target
# print(x.shape, y.shape)   #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True))   #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))


##########
# from tensorflow.keras.utils import to_categorical    
# y= to_categorical(y)
# print(y)
# print(y.shape)  

# ########### 1.keras to_categorical
# from tensorflow.keras.utils import to_categorical 
# y= to_categorical(y)    #원핫인코딩
# y= np.delete(y, 0, axis=1)   #(581012, 7)

# ############ 2. pandas get_dummies
# import pandas as pd     
# y = pd.get_dummies(y) 
# y= y.values     #  or y= y.to_numpy()  

############### 3. onehotencoder
# from sklearn.preprocessing import OneHotEncoder      #preprocessing :전처리  
# print(y.shape)   #(581012,)
# y= y.reshape(581012,1)  #1D 벡터 형태를 2D로 shape를 바꿈.
# ohe =OneHotEncoder() 
# # print(y.shape)
# y= ohe.fit_transform(y)   # ohe.fit(y) / y= ohe.transform(y)
# y= y.toarray()
# print(type(y))

x_train, x_test, y_train, y_test= train_test_split(
    x, y, shuffle=True,  
    stratify=y,     
    random_state=333,
    test_size=0.2
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler= MinMaxScaler()
# scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
# ->  x_train= scaler.fit_transform(x_train)    #x범위 만큼의 가중치 생성

x_test= scaler.transform(x_test)

print(x_train.shape, x_test.shape)      #(464809, 54) (116203, 54)

x_train= x_train.reshape(464809,54,1)
x_test= x_test.reshape(116203,54,1)


#2. 모델구성

model= Sequential()
model.add(LSTM(units = 20, input_shape= (54,1),  activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation= 'relu'))  
model.add(Dropout(0.5))
model.add(Dense(100, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation= 'softmax'))
model.summary()

#3. 컴파일, 훈련         
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping
earlystopping= EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)  


model.fit(x_train, y_train, epochs=100, batch_size=30, 
            validation_split=0.2, verbose=2, callbacks=[earlystopping])

#4. 평가, 예측
loss, accuracy= model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict= model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1)    
print("y_pred(예측값) : ",y_predict)

y_test= np.argmax(y_test, axis=1)
print("y_test(원래값) : ",y_test)  

acc= accuracy_score(y_test, y_predict)
print("accuracy_score : ",acc)

'''
loss :  0.33543387055397034
accuracy :  0.8613202571868896
y_pred(예측값) :  [1 6 0 ... 1 1 0]
y_test(원래값) :  [1 6 4 ... 1 1 0]
accuracy_score :  0.8613202757243789
'''
