import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split


a = np.array(range(1, 101)) #훈련데이터

timesteps = 5   # x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa = []        #빈 리스트를 만들어 줌.
    for i in range(len(dataset) - timesteps + 1):       #데이터의 길이 10 - timestep 5 + 1 만큼 반복(총량) =3번
        subset = dataset[i : (i + timesteps)]   #a 의 [0번째 : 0 + timestep 5] 이 subset에 들어감. > 첫번째 리스트
        aaa.append(subset)  #빈 리스트에 넣어줌.
    return np.array(aaa) 

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) #(96, 4) (96,)




x_predict = np.array(range(96, 106))    # 예상 y = range(100,107)

timesteps2 = 4   # x는 4개, y는 예측해야 하니까 만들 필요 없음.

def split_x(x_predict, timesteps2):
    ccc = []        #빈 리스트를 만들어 줌.
    for i in range(len(x_predict) - timesteps2 + 1):     
        subset = x_predict[i : (i + timesteps2)]  
        ccc.append(subset)  #빈 리스트에 넣어줌.
    return np.array(ccc) 

x_predict = split_x(x_predict, timesteps2)
print(x_predict)
print(x_predict.shape)    #(7, 4)

# x_p = ddd[:, :-1]
# y_p = ddd[:, -1]
# print(x_p, y_p)
# print(x_p.shape, y_p.shape) #(6, 4) (6,)



x_train, x_test, y_train, y_test= train_test_split(
    x, y, shuffle=True, 
    random_state=333)  #디폴트는 train_size = 0.75
print(y_train)
print(y_test)


from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #(72, 4) (24, 4) (72,) (24,)

x_train = x_train.reshape(72,4,1)
x_test = x_test.reshape(24,4,1)

x_predict = x_predict.reshape(7,4,1)

#2. 모델구성
model = Sequential()
model.add(LSTM(units= 64, input_shape=(4,1), activation='relu'))  
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))   
model.add(Dense(60, activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(1))     #flatten 필요없음.

model.summary()

#3. 컴파일, 훈련         
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping= EarlyStopping(monitor='val_loss',  
                             mode=min, patience=10, restore_best_weights=True, verbose=1)

model.fit(x, y, epochs=100, batch_size=1, 
            validation_split=0.1, verbose=2, callbacks= [earlystopping])

#4. 평가, 예측
loss= model.evaluate(x_test,y_test)
print('loss : ', loss)

result = model.predict(x_predict)
print('y의 예상값 : ', result)  

'''
loss :  [3531.537109375, 0.0]
y의 예상값 :  [[93.379906]
 [94.32136 ]
 [95.26302 ]
 [96.204895]
 [97.14696 ]
 [98.089226]
 [99.03164 ]]

'''
