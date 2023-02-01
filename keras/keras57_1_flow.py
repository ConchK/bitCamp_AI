#53-1

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()
augument_size = 100

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
x_data = train_datagen.flow(   #이미 수치화 되어있음.
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),  #x  -1은 전체 데이터==100
    np.zeros(augument_size),       #y
    batch_size=augument_size,
    shuffle=True)
    # Found 160 images belonging to 2 classes.



print(type(x_data[0])) # <class 'tuple'> // tuple : list 와 같은 형태
print(type(x_data[0][0])) # <class 'numpy.ndarray'>
print(type(x_data[0][1])) # <class 'numpy.ndarray'>


import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(49) :
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap= 'gray')

plt.show()