import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input,Flatten,Dense,Dropout,Activation,Lambda
from keras.models import Model,Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam
from decimal import *

def read_csv():
    center = 0;
    left = 0;
    right = 0;
    image_list=[]
    steering_list=[]
    steering_offset=0.22
    with open('driving_log.csv', mode='r') as infile:
        reader=csv.reader(infile)
        for row in reader:
            angle=float(row[3])
            angle=round(Decimal(angle),4)
            if (angle == 0):
                if (np.random.rand(1) > 0.20):
                    continue
            image_list.append(row[0])
            steering_list.append(float(angle))
            rand=np.random.randint(2)
            if rand==0:
                image_list.append(row[1])
                steering_list.append(float(angle) + steering_offset)
            if rand==1:
                image_list.append(row[2])
                steering_list.append(float(angle) - steering_offset)
    print("No of Samples from CSV=",len(image_list))
    print("Sample source: Center :",center, " Left :",left, " Right :",right)
    image_list=np.asarray(image_list)
    steering_list=np.asarray(steering_list)
    image_list,steering_list=shuffle(image_list,steering_list)
    return np.asarray(image_list),np.asarray(steering_list)

def limit_steering_angle(angle):
    if angle>1:
        return 1
    if angle<-1:
        return -1
    return angle

def flip_image(img):
    return cv2.flip(img,1)

def change_brightness(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand_brightness = 0.2 + np.random.uniform(0.2, 0.6)
    img[:, :, 2] = img[:, :, 2] * rand_brightness
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def shift_image_horizontal(image, steer,shift):
    shift_per_pixel=0.0025
    # Translation
    trans_matrix=np.float32([[1,0,shift],[0,1,0]])
    steer=steer + shift*shift_per_pixel
    w,h,_ = image.shape
    image=cv2.warpAffine(image, trans_matrix, (h, w))
    return image, steer

def resize_image(img):
    return cv2.resize(img,(64,64), interpolation=cv2.INTER_AREA)


def crop_image(img):
    return img[70:140,0:320]

def normalize(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


def preprocess(img):
    img = crop_image(img)
    img = resize_image(img)
    return img

def generate_images(img,steering_angle,aug_type):
    if aug_type=="none":
        return img,steering_angle
    if aug_type=="brighten":
        return change_brightness(img),steering_angle
    if aug_type=="flip":
        rand=np.random.randint(2)
        return flip_image(img),-1*steering_angle
    if aug_type=="shift":
        pixel = np.random.randint(-30,40)
        return shift_image_horizontal(img,steering_angle,pixel)

def data_generator_validation(images,angles,batch_size,max_sample_size):
    sample_count=0
    X = [];y = []
    while True:
        j=0
        while j < len(images):
            img = plt.imread(images[j].strip())
            img=preprocess(img)
            steering_angle = angles[j]
            j=j+1
            X.append(img)
            y.append(steering_angle)
            sample_count+=1
            if len(X) >= batch_size:
                yield (np.asarray(X), np.asarray(y))
                X.clear();y.clear()
            if sample_count>=max_sample_size:
                sample_count=0;j=0

def data_generator_training(images,angles,batch_size,max_sample_size):
    sample_count=0
    aug_list=['flip','brighten','shift']
    X=[];y=[]
    while True:
        j=0
        while j <len(images):
            img=plt.imread(images[j].strip())
            steering_angle=angles[j]
            img=preprocess(img)
            j=j+1
            if(steering_angle==0):
                for aug_type in aug_list:
                    aug_img,angle=generate_images(img,steering_angle,aug_type)
                X.append(aug_img)
                y.append(limit_steering_angle(angle))
                sample_count+=1
            else:
                X.append(img)
                y.append(limit_steering_angle(steering_angle))
                sample_count += 1
            if sample_count>=max_sample_size:
                print("End of Epoch. Reset Image index to 0. Sample Count=",sample_count)
                sample_count=0;j=0
                yield (np.asarray(X), np.asarray(y))
                X.clear();y.clear()
                break
            if len(X)>=batch_size:
                yield(np.asarray(X[0:batch_size]), np.asarray(y[0:batch_size]))
                X.clear();y.clear()

def covnet(shape):
    model = Sequential()
    model.add(Lambda(lambda x:x/127.5 -1.0, input_shape=shape))
    model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='FC2'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='FC3'))
    model.add(Dense(1))
    model.compile('adam', 'mean_squared_error', metrics=['mean_squared_error'])
    return model

def visualize_data(image_array,steering_array):
    train_angles = np.unique(steering_array)
    hist_x = []
    hist_y = []
    for i in range(len(train_angles)):
        angle = train_angles[i]
        hist_x.append(float(angle))
        sample = np.where(steering_array == angle)
        sample = np.asarray(sample).flatten()
        print("angle=", angle, " count= ", len(sample), sample)
        count = len(sample)
        hist_y.append(count)
    hist_x = np.asarray(hist_x)
    hist_y = np.asarray(hist_y)
    plt.bar(hist_x, hist_y, 1 / len(hist_x))
    plt.title('After augmentation and zero bias correction')
    plt.xlabel('angles')
    plt.ylabel('frequency')
    plt.show()


def get_train_validation_datasets():
    image_array,steering_array=read_csv()
    image_array,steering_array=shuffle(image_array,steering_array)
    return train_test_split(image_array,steering_array,test_size=0.2, random_state=45)


def train_network():
    X_train,X_val,y_train,y_val=get_train_validation_datasets()
    no_of_zero_samples=len(np.asarray(np.where(y_train == 0)).flatten())
    print("No of zero elements=", no_of_zero_samples)
    batch_size =256
    data_samples=len(X_train) + no_of_zero_samples
    max_train_samples = int(data_samples/batch_size)*batch_size
    number_of_epoch=15
    max_validation_samples=len(X_val)
    print("Max train samples=", max_train_samples)
    print("Max val samples =", max_validation_samples)
    model=covnet((64,64,3))
    history = model.fit_generator(data_generator_training(X_train,y_train,batch_size,max_train_samples),max_train_samples,number_of_epoch,verbose=1,validation_data=data_generator_validation(X_val,y_val,batch_size,max_validation_samples),nb_val_samples=max_validation_samples)


import os

def main(_):
    train_network()




if __name__ == '__main__':
    tf.app.run()




