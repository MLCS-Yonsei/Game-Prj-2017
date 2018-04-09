from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import theano
import os
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
#load data
train_x = np.load('./data/total_train.npz')['a']
ipt=np.rollaxis(np.rollaxis(train_x,2,0),2,0)


img_rows,img_cols,img_depth=120,160,10
X_tr=[]  



for i in range(150):
    X_tr.append(ipt[:,:,i*10:(i+1)*10])

X_tr_array = np.array(X_tr)

num_samples = len(X_tr_array) 
# print(len(X_tr))
# print(X_tr[0].shape)

#assign label
label=np.ones((num_samples,),dtype = int)
label[0:25]= 0
label[25:50] = 1
label[50:75] = 2
label[75:100] = 3
label[100:125]= 4
label[125:] = 5

train_data = [X_tr_array,label]

(X_train, y_train) = (train_data[0],train_data[1])

train_set = np.zeros((num_samples, 1, img_rows,img_cols,img_depth))

for h in range(num_samples):
    train_set[h][0][:][:][:]=X_train[h,:,:,:]
 

patch_size = 10   # img_depth or number of frames used for each video

print(train_set.shape, 'train samples')

# CNN Training parameters

batch_size = 2
nb_classes = 6
nb_epoch =50

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)


# number of convolutional filters to use at each layer
nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5,5]

# Pre-processing

train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)


'''
listing = os.listdir('./dataset/boxing')[:1]

for vid in listing:
    vid = './dataset/boxing/'+vid
    frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(5)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
  

    for k in range(15):
        ret, frame = cap.read()
        # frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray)
        # plt.show()
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    input=np.array(frames)

    print (input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print (ipt.shape)

    X_tr.append(ipt)

print(X_tr[0].shape)
# plt.imshow(gray)
#     plt.show()
#     frames.append(gray)

'''