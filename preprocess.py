import cv2
import numpy as np
import os

def pre_process(filepath,kernel) :
  img = cv2.imread(filepath,0) #0 for grayscale
  resized = cv2.resize(img,(218,192),interpolation=cv2.INTER_AREA) #resizing to max size
  dst = cv2.fastNlMeansDenoising(resized,40,7,21) #removing the noise
  blur = cv2.GaussianBlur(dst,(5,5),0)  #adding gaussian blur to make in white and black
  ret,thr = cv2.threshold(dst,0,225,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #thr returns thresholding image after binary filter
  open1 = cv2.morphologyEx(thr,cv2.MORPH_OPEN,kernel)  #opening in  morphing to remove background noise
  close1 = cv2.morphologyEx(open1,cv2.MORPH_CLOSE,kernel) #closing remove small holes in foreground objects
  resizedd = cv2.resize(close1,(112,112))
  final_image = resizedd.reshape((112,112,1))
  return final_image

  
def main() :
  directory = '/Users/silpasoni/Desktop/3-2/ml_project/processed_images/'
 # finalpath = '/Users/silpasoni/Desktop/3-2/ml_project/processed_images/'
  kernel = np.ones((5,5),np.uint8)
  for filename in os.listdir(directory) :    
    filepath = os.path.join(directory,filename)
    image = pre_process(filepath,kernel)
    dstpath = os.path.join(directory,filename)
    print('Done' + filename)
    cv2.imwrite(dstpath,image)
    


if __name__ == '__main__' :
    main()
