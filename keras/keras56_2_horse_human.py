#수치화, numpy 저장
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 이미지를 변환하고 증폭시키는 역할
train_datagen = ImageDataGenerator(
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


train_gen = train_datagen.flow_from_directory( # classes 설정을 생략하면 폴더의 순서(오름차순)로 label을 결정
    './_data/horse-or-human/', 
    target_size=(200,200), #크기에 상관없이 200, 200 으로 압축
    batch_size=100000, #정확한 사이즈를 모르면 그냥 큰 숫자
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
    # Found 2520 images belonging to 3 classes.
) 

validation_gen = train_datagen.flow_from_directory( # classes 설정을 생략하면 폴더의 순서(오름차순)로 label을 결정
    './_data/horse-or-human/', 
    target_size=(200,200), #크기에 상관없이 200, 200 으로 압축
    batch_size=100000, #정확한 사이즈를 모르면 그냥 큰 숫자
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
    # Found 2520 images belonging to 3 classes.
) 


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model =  Sequential()                   #target size, color 데이터= 3
model.add(Conv2D(64, (2, 2), input_shape=(200, 200, 3), activation='relu'))  #활성화 함수는 relu
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(32, (2, 2), activation='relu', padding= 'same'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(16, (2, 2), activation='relu', padding= 'same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(120, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping= EarlyStopping(monitor='val_loss', mode=min, patience=10, restore_best_weights=True, verbose=1)

hist = model.fit(train_gen,
                   # batch_size=50,
                   steps_per_epoch=16, 
                   epochs=100, 
                   validation_data= validation_gen, 
                   validation_steps= 300,
                   # validation_split= 0.2, 
                   callbacks=[earlystopping], verbose=2 )  # train data에서 나눔. 


#4. 평가, 예측
loss=model.evaluate(train_gen)  #x_test,y_test 값으로 평가
print('loss :', loss)
y_predict=model.predict(train_gen)  #x_test값으로 y_predict 예측
# print('x_test :', x_test)
# print('y_predict :', y_predict)
