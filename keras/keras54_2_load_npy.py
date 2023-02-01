import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# np.save('./_data/brain/brain_x_tain.npy', arr=xy_train[0][0])
# np.save('./_data/brain/brain_y_tain.npy', arr=xy_train[0][1])

# # np.save('./_data/brain/brain_xy_tain.npy', arr=xy_train[0])  # tuple 사용하면 가능./ 나중에 분리해서 사용

# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])

x_train = np.load('./_data/brain/brain_x_tain.npy')
y_train = np.load('./_data/brain/brain_y_tain.npy')
x_test = np.load('./_data/brain/brain_x_test.npy')
y_test = np.load('./_data/brain/brain_y_test.npy')

print(x_train.shape, y_train.shape)  #(160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)  #(120, 200, 200, 1) (120,)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model =  Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(200, 200, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit ( x_train, y_train,
                    batch_size=16,
                    #steps_per_epoch=16, 
                    epochs=100, 
                    # validation_data=(xy_test[0][0], xy_test[0][1]),  #numpy 저장 때 이미 나눴기 때문에 필요x
                    #validation_steps=4,
                    validation_split= 0.2 )  # train data에서 나눔. 



accuracy = hist.history['acc'] 

val_acc = hist.history['val_acc'] 
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# [-1] 모든 훈련에 관한 loss 값이 나오기에 맨 마지막 훈련의 값을 출력 
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])
print('loss : ', loss[-1]) 
print('val_loss : ', val_loss[-1])

'''
accuracy :  1.0
val_acc :  1.0
loss :  0.20784081518650055
val_loss :  0.18249444663524628
'''

