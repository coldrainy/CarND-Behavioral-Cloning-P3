import numpy as np
import csv


data=[]
filepath = 'data/data/driving_log.csv'
with open(filepath) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        row[0] = 'data/data/IMG/'+row[0].split('/')[-1]
        row[1] = 'data/data/IMG/'+row[1].split('/')[-1]
        row[2] = 'data/data/IMG/'+row[2].split('/')[-1]
        data.append(row)
# folder_names = ['my_data1/','my_data2/','my_data_3/',
        # 'my_data4/','my_data5/','my_data6/','my_data7/']
folder_names = ['my_data1/']
file_paths = []
for folder_name in folder_names:
    file_paths.append(folder_name+'driving_log.csv')
for file_path in file_paths:
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

from sklearn.model_selection import train_test_split
train_samples,validation_samples = train_test_split(data,test_size=0.2)



import cv2
import sklearn
from numpy import array



def generator(samples,batch_size=32):
    num = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0,num,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steerings = []
            rectify = 0.2
            for sample in batch_samples:
                name_center = sample[0]
                name_left = sample[1]
                name_right = sample[2]
                image_center = cv2.imread(name_center)
                image_center_flipped = np.fliplr(image_center)
                image_left = cv2.imread(name_left)
                image_left_flipped = np.fliplr(image_left)
                image_right = cv2.imread(name_right)
                image_right_flipped = np.fliplr(image_right)
                steering_center = float(sample[3])
                steering_center_flipped = -steering_center
                steering_left = float(sample[3])+rectify
                steering_left_flipped = -steering_left
                steering_right = float(sample[3])-rectify
                steering_right_flipped = -steering_right
                images.append(image_center)
                images.append(image_center_flipped)
                images.append(image_left)
                images.append(image_left_flipped)
                images.append(image_right)
                images.append(image_right_flipped)
                steerings.append(steering_center)
                steerings.append(steering_center_flipped)
                steerings.append(steering_left)
                steerings.append(steering_left_flipped)
                steerings.append(steering_right)
                steerings.append(steering_right_flipped)

            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train,y_train)

train_generator = generator(train_samples,batch_size=60)
validation_generator = generator(validation_samples,batch_size=60)

from keras.models import Sequential
from keras.layers import Lambda,Flatten,Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import tensorflow as tf

model = Sequential()
model.add(Lambda(lambda x:x/255-0.5,
                    input_shape=(160,320,3),
                    output_shape=(160,320,3)))

model.add(Cropping2D(cropping=((50,20),(0,0))))

##########LeNet############
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))


model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer = 'adam')
history_object=model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*3*2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*3*2,
                    nb_epoch=10,verbose = 1)
model.save('model.h5')


import matplotlib.pyplot as plt
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

