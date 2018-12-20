import numpy as np
import cv2
import os

def get_images(directory) :
  images = []
  image_names = []
  for filename in os.listdir(directory) :
    img = cv2.imread(directory + '/' + filename ,cv2.IMREAD_GRAYSCALE)
    images.append(img)
    image_names.append(filename)
  return images,image_names

def processing(images,image_names) :
  #added all the name of images into names array as a pair of unicodes
  names = []
  for image_name in image_names :
    name = image_name[:-4]
    name = name.split("_")
    name = name[3:]
    names.append(name)
  #classes is a list of unicodes
  classes = set([])
  for name in names :
    for a in name :
      classes.add(int(a))
  classes = list(classes)
  #class_map is map with unicode and a class
  class_map = {}
  i = 0
  for a in classes :
    class_map[a] = i
    i = i+1
  #pair_classes is a list of pair of unicode classes similar to names
  pair_classes = []
  for name in names :
    cl = []
    for a in name :
      cl.append(class_map[int(a)])
    pair_classes.append(cl)
  #y is array for each image as 1s mapped to thier class in 80 zeros array
  y = np.zeros((len(images),80))
  for i in range(len(y)) :
    y[i][pair_classes[i]] = 1

  return y
    
    
def mapping() :
  training_data = '/Users/silpasoni/Desktop/3-2/ml_project/processed_images/'
  cross_validation = '/Users/silpasoni/Desktop/3-2/ml_project/cross_validation_set/'
  train_img, train_names = get_images(training_data)
  test_img, test_names = get_images(cross_validation)
  y_train = processing(train_img,train_names)
  y_test = processing(test_img,test_names)
  x_train = np.array(train_img)
  x_test = np.array(test_img)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
  return x_train,y_train,x_test,y_test

def main() :
  x_train, y_train, x_test, y_test = mapping()
  print(y_train.shape)
  print(y_test.shape)
  
if __name__ == "__main__" :
    main()
  
