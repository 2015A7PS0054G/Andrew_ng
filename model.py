from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout
from mapping import mapping
import numpy as np
import os
import cv2
import glob
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Model

x_train, y_train, x_test, y_test = mapping()
x_train = x_train[:,:,:,None]
x_test = x_test[:,:,:,None]
#reshaping images into 4D
x_train = x_train.reshape((-1,112,112,1)).astype(np.float32)
#y_train = (np.arange(80)==y_train[:,None]).astype(np.float32)
print(x_train.shape)
x_test = x_test.reshape((-1,112,112,1)).astype(np.float32)
#y_test = (np.arange(80)==y_test[:,None]).astype(np.float32)
#sequential model
model = Sequential()
#32 channels and 5*5 sliding window
model.add(Convolution2D(32,(5,5),activation = 'relu',input_shape = (112,112,1)))
#max pooling with 2*2 filters
model.add(MaxPooling2D(pool_size = (2,2)))
#32 channels and 4*4 sliding window
model.add(Convolution2D(32,(4,4),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
#64 channels and 3*3 sliding window
model.add(Convolution2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#128 channels and 3*3 sliding window
model.add(Convolution2D(128,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#flattening the output
model.add(Flatten())
model.add(Dense(512,activation='relu',input_shape=(112,112,1)))
model.add(Dropout(0.25))
model.add(Dense(80,activation='sigmoid',input_shape=(112,112,1)))
#compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print("compiled")
#one hot encode outputs
#y_train = np_utils.to_categorical(y_train,num_classes = 80)
print(y_train.shape)
#y_test = np_utils.to_categorical(y_test,num_classes = 80)
#fit the model
model.fit(x_train,y_train,batch_size=32,epochs=50,verbose=1)
#testing on cross_validation data
score = model.evaluate(x_test,y_test,verbose=1)
print('test loss ',score[0])
print('test accuracy',score[1])

def predict(i):
    X_cross = list()
    # Y_cross = list()
    
    image = cv2.imread(i)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(image,7) 
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = cv2.resize(otsu, (112,112))
    
    img = img.reshape(112,112,1)
    X_cross.append(img)
    

    #Y_cross = np.asarray(Y_cross)
    X_cross = np.asarray(X_cross)/255

    Y_p = model.predict(X_cross)[0]

    val1 = np.argmax(Y_p)
    s1 = Y_p[val1]
    Y_p[val1] = 0

    val2 = np.argmax(Y_p)
    s2 = Y_p[val2]
    Y_p[val2] = 0

    val3 = np.argmax(Y_p)
    s3 = Y_p[val3]
    Y_p[val3] = 0

    val4 = np.argmax(Y_p)
    s4 = Y_p[val4]
    Y_p[val4] = 0

    val5 = np.argmax(Y_p)
    s5 = Y_p[val5]

    s2 = s2/s1
    s3 = s3/s1
    s4 = s4/s1
    s5 = s5/s1
    s1 = 1

    val = [val1, val2 , val3 , val4 , val5]
    li =  [s1,s2,s3,s4,s5]

    #print(y_p)
    thelist = []
    for nn in range(5):
        if li[nn] > 0.5:
            gh = val[nn] +2304
            thelist.append(gh)

    thelist = np.sort(thelist)        
    return thelist  

for i in glob.glob('./*.png'):
    print(predict(i))
    print(i)
