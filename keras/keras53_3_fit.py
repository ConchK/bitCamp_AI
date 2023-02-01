import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지를 변환하고 증폭시키는 역할
xy_train = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 이미지 수평으로 
    vertical_flip=True,
    width_shift_range=0.1, #이동 
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest' # 가까이 있는 것으로 채움
)

xy_test= ImageDataGenerator(
      rescale=1./255
)
                        #파일
xy_train = xy_train.flow_from_directory( # 안에 있는 ad, normal 은 0, 1로 인식
    './_data/brain/train/', 
    target_size=(100,100), #크기에 상관없이 200, 200 을 압축
    batch_size=1000000,   #데이터 크기가 파악되면 그대로 적고, 아님 그냥 크게 하나로 잡아 줌.
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 160 images belonging to 2 classes.
) 

xy_test = xy_test.flow_from_directory( # 안에 있는 ad, normal 은 0, 1로 인식
    './_data/brain/test/', 
    target_size=(200,200), #크기에 상관없이 200, 200 을 압축
    batch_size=10,
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 160 images belonging to 2 classes.
) 


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model =  Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

                # 전체 x 값      # 전체 y 값
hist = model.fit(xy_train[0][0], xy_train[0][1],
                    batch_size=16,
                    #steps_per_epoch=16, 
                    epochs=100, 
                    validation_data=(xy_test[0][0], xy_test[0][1]), #validateion_split  사용가능.
                    #validation_steps=4,
                    )



accuracy = hist.history['acc'] 

val_acc = hist.history['val_acc'] 
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# [-1] 모든 훈련에 관한 loss 값이 나오기에 맨 마지막 훈련의 값을 출력 
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])
print('loss : ', loss[-1]) 
print('val_loss : ', val_loss[-1])


# 그림 그리삼



batch = xy_train.next()

print(batch)
print(len(batch)) #2
print(type(batch)) #tuple

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for i in range(4) :
    plt.subplot(2, 2, i+1)
    plt.imshow(batch[0][i])
plt.show()