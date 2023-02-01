import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 데이터를 변환하고 증폭
xy_train = ImageDataGenerator(
    rescale=1./255, #행렬의 값이 0과 1사이조 재조정
    horizontal_flip=True, # 이미지 수평으로 
    vertical_flip=True,
    width_shift_range=0.1, # 일정 범위 내에서 무작위로 가로로 이동 
    height_shift_range=0.1,
    rotation_range=5,   # 5도 임의 회전
    zoom_range=1.2, #0.8~1.2 임의로 줌
    shear_range=0.7,    # 0.7강도
    fill_mode='nearest' # 가까이 있는 것으로 채움
)

xy_test= ImageDataGenerator(
      rescale=1./255
)
                        #파일
xy_train = xy_train.flow_from_directory(  # classes 설정을 생략하면 폴더의 순서(오름차순)로 label을 결정/ 0과 1
    './_data/brain/train/', 
    target_size=(100,100), #크기에 상관없이 200, 200 으로 압축
    batch_size=10,
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 160 images belonging to 2 classes.
) 

                     #파일
xy_test = xy_test.flow_from_directory( # 안에 있는 ad, normal 은 0, 1로 인식
    './_data/brain/test/', 
    target_size=(100,100), #크기에 상관없이 200, 200 을 압축
    batch_size=10,  #batch_size를 최대한 늘려서 데이터를 한번에 뽑아낼 수 있음
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 120 images belonging to 2 classes.
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
# model.add(Dense(2, activation='softmax'))  가능.
# 대신 loss='sparse_categorical_crossentropy'

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, 
                    steps_per_epoch=16, #160개의 데이터를 10으로 나눠줌. 1epoch에 들어가는 steps
                    epochs=10, 
                    validation_data=xy_test, 
                    validation_steps=4,     #찾아봐-
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