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
    target_size=(200,200), #크기에 상관없이 200, 200 을 압축
    batch_size=10,
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 160 images belonging to 2 classes.
) 

                     #파일
xy_test = xy_test.flow_from_directory( # 안에 있는 ad, normal 은 0, 1로 인식
    './_data/brain/test/', 
    target_size=(200,200), #크기에 상관없이 200, 200 을 압축
    batch_size=10,  #batch_size를 최대한 늘려서 데이터를 한번에 뽑아낼 수 있음
    class_mode='binary', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 120 images belonging to 2 classes.
) 

print(xy_train)
# <keras.preprocessing.image.ImageDataGenerator object at 0x0000012E32072FA0>

#리스트 - 데이터의 형태는 상관 없음.
#튜플 - 소괄()호 형태로 모여 있고 그 자체를 변경할 수 없다.
#딕셔너리 - key(객체), value(가진 값)의 형태 {}

# from sklearn.datasets import load_iris
# datasets = load_iris()

# print(datasets)
# print(xy_train[0])
# error
# print(xy_train[16][0].shape) #(5, 200, 200, 1)
# print(xy_train[16][1].shape) #(5, 200, 200, 1)
# ValueError: Asked to retrieve element 16, but the Sequence has length 16   

# print(xy_train[0])
# print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[15][0].shape) #(5, 200, 200, 1)
print(xy_train[15][1].shape) #(5, 200, 200, 1)

print(type(xy_train))  # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> // tuple : list 와 같은 형태
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>