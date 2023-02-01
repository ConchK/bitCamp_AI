#수치화, numpy 저장
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


                    #파일
train_generator = train_datagen.flow_from_directory( # classes 설정을 생략하면 폴더의 순서(오름차순)로 label을 결정
    './_data/cat_dog/train/', 
    target_size=(200,200), #크기에 상관없이 200, 200 으로 압축
    batch_size=100000, #정확한 사이즈를 모르면 그냥 큰 숫자
    class_mode='binary', # 이진분류이기 때문에
    color_mode='rgb',
    shuffle=True
    # Found 25000 images belonging to 2 classes.
) 

validation_generator = train_datagen.flow_from_directory( # classes 설정을 생략하면 폴더의 순서(오름차순)로 label을 결정
    './_data/cat_dog/train/', 
    target_size=(200,200), #크기에 상관없이 200, 200 으로 압축
    batch_size=100000,
    class_mode='binary', # 이진분류
    shuffle=True
    # Found 0 images belonging to 0 classes.
) 

# import matplotlib.pyplot as plt

# sample_training_images, _ = next(xy_train)

# # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
    
# plotImages(sample_training_images[:5])


# <keras.preprocessing.image.DirectoryIterator object at 0x000002E99B59A970>



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
# print(xy_train[0][1])
# print(xy_train[0][0].shape) #
# print(xy_train[0][1].shape) #

np.save('./_data/cat_dog/cat_dog_x_train.npy', arr=train_generator[0][0])
np.save('./_data/cat_dog/cat_dog_y_train.npy', arr=train_generator[0][1])

np.save('./_data/cat_dog/cat_dog_x_val.npy', arr=validation_generator[0][0])
np.save('./_data/cat_dog/cat_dog_y_val.npy', arr=validation_generator[0][1])


# np.save('./_data/brain/brain_xy_tain.npy', arr=xy_train[0])  # tuple 사용하면 가능./ 나중에 분리해서 사용

# np.save('./_data/brain/cat_dog_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/cat_dog_y_test.npy', arr=xy_test[0][1])


# print(type(xy_train))  # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'> // tuple : list 와 같은 형태
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

