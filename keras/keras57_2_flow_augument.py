#53-1

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()
augument_size = 40000               #60000
randidx = np.random.randint(x_train.shape[0], size= augument_size )

print(randidx)      #[19820 49782 13610 ... 57374  7195 39992]
print(len(randidx))   #40000  40000개를 랜덤으로 뽑음.

x_augument = x_train[randidx].copy()   #데이터의 원본은 두고 카피본에 추출.
y_augument = y_train[randidx].copy() 
print(x_augument.shape, y_augument.shape)       #(40000, 28, 28) (40000,)   아직 증폭x, 추출만 함.

x_augument = x_augument.reshape(40000,28,28,1)


# 이미지를 변환하고 증폭시키는 역할
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 이미지 수평으로 
    # vertical_flip=True,
    width_shift_range=0.1, #이동 
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest' # 가까이 있는 것으로 채움
)

                        #파일
x_augumented = train_datagen.flow(   #이미 수치화 되어있음.
    x_augument,
    y_augument,  
    batch_size=augument_size,
    shuffle=True)
    # Found 160 images belonging to 2 classes.

print(x_augumented[0][0].shape)  #(40000, 28, 28, 1)
print(x_augumented[0][1].shape)  #(40000,)

x_train = x_train.reshape(60000,28,28,1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))

print(x_train.shape, y_train.shape)  #(100000, 28, 28, 1) (100000,)


# print(type(x_data[0])) # <class 'tuple'> // tuple : list 와 같은 형태
# print(type(x_data[0][0])) # <class 'numpy.ndarray'>
# print(type(x_data[0][1])) # <class 'numpy.ndarray'>


# import matplotlib.pyplot as plt
# plt.figure(figsize=(7, 7))
# for i in range(49) :
#     plt.subplot(7, 7, i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][0][i], cmap= 'gray')

# plt.show()