import numpy as np
from matplotlib import pyplot as plt
import cv2
import copy
import keras
import tensorflow as tf
import os
from random import shuffle 
from tqdm import tqdm 
from keras.models import load_model
import shutil

#IF WINDOWS OS THEN USE tkinter 
from tkinter import filedialog
from tkinter import *

#Import model
model = load_model("./keras model/septinmodel.h5")

#Load Image Directory
path = filedialog.askdirectory()
#path = "./images"
goodDir = path+'/good/'
badDir = path+'/bad/'
if not os.path.exists(goodDir):
    os.mkdir(goodDir)
    print("Directory ", goodDir, " Created ")
else:    
    print("Directory " , goodDir ,  " already exists")
    print("Overwriting previous folder...")

if not os.path.exists(badDir):
    os.mkdir(badDir)
    print("Directory ", badDir, " Created ")
else:    
    print("Directory " , badDir ,  " already exists")
    print("Overwriting previous folder...")

#Use Model to predict

for i in os.listdir(path):
    if i.endswith('.tif'):
        imgPath = os.path.join(path, i)
        imgData = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
        imgData = cv2.resize(imgData,(100,100))
        imgData = np.array(imgData).reshape(1,100,100,1)
        model_out=model.predict([imgData])
        
        if np.argmax(model_out)==1:
            shutil.move(imgPath, os.path.join(goodDir, i))
        else:
            shutil.move(imgPath, os.path.join(badDir, i))
    else:
        pass

 

    



