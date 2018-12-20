import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.optimizers import Adam
from keras import losses
import glob

def load():
    return load_model("model_json")

model = load()
def predict(i):
    X_cross = list()
    # Y_cross = list()
    
    image = cv2.imread(i)
    image = cv2.resize(image,(218,192),interpolation=cv2.INTER_AREA)
    image = cv2.fastNlMeansDenoising(image,40,7,21)
    blur = cv2.GaussianBlur(image,(5,5),0) 
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    open1 = cv2.morphologyEx(otsu,cv2.MORPH_OPEN,kernel)
    close1 = cv2.morphologyEx(open1,cv2.MORPH_CLOSE,kernel)
    img = cv2.resize(close1,(112,112)) 
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
