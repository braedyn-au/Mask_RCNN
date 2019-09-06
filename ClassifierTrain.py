#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import cv2
import copy
import keras
import tensorflow as tf
import os
from random import shuffle 
from tqdm import tqdm 
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


#load data, probably best to put the script in the folder with data
train_data = './Classifier Images/'
# test_data = './test/'


# In[168]:


#GOOD = [1,0]
#BAD = [0,1]
def train_data_with_label():
    '''
    labels each image in the train folder
    '''
    train_images =[]

    good = train_data + 'good'
    bad = train_data + 'bad'
    ngood = 0
    nbad = 0
    for i in os.listdir(good):
        if i.endswith('.DS_Store'):
            pass
        else:
            path = os.path.join(good, i)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(100,100))
            train_images.append([np.array(img), 1])
            ngood = ngood + 1
            #print(i)
    for i in os.listdir(bad):
        if i.endswith('.DS_Store'):
            pass
        else:
            path = os.path.join(bad,i)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(100,100))
            train_images.append([np.array(img),0])
            nbad = nbad + 1
            #print(i)
    # repeat good samples due to lack of data
    if nbad > ngood:
        n = round(nbad/ngood)
        good_data = train_images[:ngood]
        good_data = good_data*(n-1)
        train_images = train_images + good_data
    shuffle(train_images)
    return train_images


# In[169]:


def test_data_with_label():
    '''
    Unused
    labels each image in the test folder
    not used
    '''
    test_images =[]

    good = test_data + 'good'
    bad = test_data + 'bad'
    
    for i in os.listdir(good):
        if i.endswith('.DS_Store'):
            pass
        else:
            path = os.path.join(good, i)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(100,100))
            test_images.append([np.array(img), 1])
            #print(i)
    for i in os.listdir(bad):
        if i.endswith('.DS_Store'):
            pass
        else:    
            path = os.path.join(bad,i)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(100,100))
            test_images.append([np.array(img),0])
            #print(i)
    shuffle(test_images)
    return test_images


# In[170]:


training_images = train_data_with_label()
shuffle(training_images)
##testing_images = test_data_with_label()


# In[187]:


tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,100,100,1)
tr_lbl_data = np.array([i[1] for i in training_images])
##tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,100,100,1)
##tst_lbl_data = np.array([i[1] for i in testing_images])


# In[188]:


#convert to one hot label
tr_lbl_data=keras.utils.to_categorical(tr_lbl_data, num_classes=2, dtype='float32')
##tst_lbl_data=keras.utils.to_categorical(tst_lbl_data, num_classes=None, dtype='float32')


# In[189]


##print(tr_lbl_data.shape,tr_img_data.shape)
##print(tr_lbl_data[4].shape)


# In[190]:


from keras.models import Sequential 
from keras.layers import *
from keras.optimizers import * 
from keras.callbacks import *


# In[196]:


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=3,strides=1,padding='same',activation='relu',input_shape=(100,100,1)))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Conv2D(filters=50,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Conv2D(filters=80,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(2,activation='softmax'))
optimizer = Adam(lr=1e-4)

model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
#model.fit(x=tr_img_data,y=tr_lbl_data,epochs=20,batch_size=5,validation_data=(tst_img_data,tst_lbl_data))
model.summary()


# In[198]:


model.fit(x=tr_img_data,y=tr_lbl_data,epochs=20,batch_size=4, validation_split=0.2)


# In[256]:


#subplot not necessary in final script
#input data in this step and add function to seperate in os
#fig = plt.figure(figsize=(14,14))
##imgs = training_images
#for count, data in enumerate(imgs):
#    y = fig.add_subplot(8,6,count+1)
#    img = data[0]
#    data = img.reshape(1,100,100,1)
#    model_out=model.predict([data])
#    if np.argmax(model_out)==1:
#        str_label='good'
#    else:
#        str_label='bad'
#    y.imshow(img,cmap='gray')
#    plt.title(str_label)
#    y.axes.get_xaxis().set_visible(False)
#    y.axes.get_yaxis().set_visible(False)

##img =  np.array([i[0] for i in imgs]).reshape(-1,100,100,1)
##lbl = np.array([i[1] for i in imgs])
##lbl = keras.utils.to_categorical(lbl, num_classes=2, dtype='float32')
score = model.evaluate(tr_img_data,tr_lbl_data,verbose=0)
print("Score: ",score[1])

#Save model
dirName = './keras model'
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")
    print("Overwriting previous model...")
model.save(dirName+"/septinmodel.h5")

# In[ ]:





# In[ ]:




