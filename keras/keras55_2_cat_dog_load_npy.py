#모델링 & 결과
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#1.데이터

# np.save('./_data/cat_dog/cat_dog_x_train.npy', arr=train_generator[0][0])
# np.save('./_data/cat_dog/cat_dog_y_train.npy', arr=train_generator[0][1])

# np.save('./_data/cat_dog/cat_dog_x_val.npy', arr=validation_generator[0][0])
# np.save('./_data/cat_dog/cat_dog_y_val.npy', arr=validation_generator[0][1])

x_train = np.load('./_data/cat_dog/cat_dog_x_train.npy')
y_train = np.load('./_data/cat_dog/cat_dog_y_train.npy')
x_val = np.load('./_data/cat_dog/cat_dog_x_val.npy')
y_val = np.load('./_data/cat_dog/cat_dog_y_val.npy')

x_train, x_test, y_train, y_test= train_test_split(
    x_train, y_train, shuffle=True,  
    random_state=333,
    test_size=0.2
)


# print(x_train.shape, y_train.shape)  #(20000, 200, 200, 3) (20000,)
# print(x_test.shape, y_test.shape)  #(5000, 200, 200, 3) (5000,)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model =  Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(200, 200, 3)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Conv2D(16, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# from tensorflow.keras.callbacks import EarlyStopping
# earlystopping= EarlyStopping(monitor='val_loss', mode=min, patience=10, restore_best_weights=True, verbose=1)

model.fit( x_train, y_train,
                    # batch_size=50,
                    steps_per_epoch=16, 
                    # epochs=100, 
                    validation_data=(x_val, y_val),  #numpy 저장 때 이미 나눴기 때문에 필요x
                    validation_steps=8,
                    # validation_split= 0.2,
                    verbose=2 )  # train data에서 나눔. 


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)  #x_test,y_test 값으로 평가
print('loss :', loss)
y_predict=model.predict(x_test)  #x_test값으로 y_predict 예측
# print('x_test :', x_test)
# print('y_predict :', y_predict)

'''

'''