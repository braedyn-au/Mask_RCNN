# detect rings in photos with mask rcnn model
from os import listdir
import os
import scipy
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
from skimage.util import pad
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn import visualize as vs
#IF WINDOWS OS THEN USE tkinter 
from tkinter import filedialog, messagebox
from tkinter import *
#from maskrcnn2 import SeptinDataset
# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "Septin_cfg"
    # number of classes (background + Septin)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1   
    DETECTION_MIN_CONFIDENCE = 0.6 

    
# plot a number of photos with ground truth and predictions
def plot_predicted(imageDirectory, outputDirectory, model, cfg, n_images=23):
    # load image and mask
    for i in listdir(imageDirectory):
        if i.endswith('.tif'):
            image_number = listdir(imageDirectory).index(i)
            # load the image and mask
            
            imgPath = os.path.join(imageDirectory, i)
            image = skimage.io.imread(imgPath)
            #image = SeptinDataset.load_Septin(imgPath)
            outPath = os.path.join(outputDirectory, i)
            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
            #mask, _ = SeptinDataset.load_mask(i)
            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, cfg)
            # convert image into one sample
            sample = expand_dims(scaled_image, 0)
            # make prediction
            yhat = model.detect(sample, verbose=0)[0]
            # define subplot
            #pyplot.subplot(n_images, 2, i*2+1)
            # plot raw pixel data
            pyplot.imshow(image)
            #pyplot.title('predcited masks')
            #pyplot.show()
            # plot masks
            mask = yhat['masks']
            for j in range(mask.shape[2]):
                pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
                pyplot.title('Predicted Mask')
            
            # get the context for drawing boxes
            #pyplot.subplot(n_images/2, 1, image_number*2+2)
            # plot raw pixel data
            #masks = yhat['masks']
            #image = vs.apply_mask(image,masks[:2],vs.random_colors(1)[0])
            

            #pyplot.imshow(image)
            pyplot.title('Predicted')
            ax = pyplot.gca()
            # plot each box
            n = 1
            print("Looking at image", i ,"...")
            for box in yhat['rois']:
                # get coordinates
                y1, x1, y2, x2 = box
                # increase box size
                x1 = x1 - 10
                y1 = y1 - 10
                y2 = y2 + 10
                x2 = x2 + 10
                # calculate width and height of the box
                width, height = x2 - x1, y2 - y1
                # create the shape
                rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                # draw the box
                ax.add_patch(rect)
                # Cut the rois from each image
                # take output directory as user input in seperate script
                img_crop = image[y1:y2,x1:x2]
                # pad to 100x100
                h, w, d = img_crop.shape
                if h < 120 and w < 120:
                    dh = (120-h)//2
                    dw = (120-w)//2
                    img_crop = pad(img_crop, ((dh,0),(dw,0),(0,0)), 'constant')
                    h, w, d = img_crop.shape
                    img_crop = pad(img_crop,((0,120-h),(0,120-w), (0,0)),'constant')
                else:
                    pass
                if img_crop.size != 0 and np.mean(img_crop) != 0:
                    skimage.io.imsave(outPath+str(n)+'.tif',img_crop, check_contrast = False)
                    print("Image Cropped ", n)
                n = n + 1
            # show the figure
            pyplot.show()
            
        
        else:
            pass

 
# load the images
messagebox.showinfo("Images", "Load the folder containing images")
imageDirectory = filedialog.askdirectory()
#train_set.load_dataset('Ring', is_train=True)
#train_set.prepare()
#print('Train: %d' % len(train_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights with tkinter
messagebox.showinfo("Model", "Load model found in septin_cfg... folder")
model_path = filedialog.askopenfilename() 
model.load_weights(model_path, by_name=True)
messagebox.showinfo("Output", "Choose output folder")
outputDirectory = filedialog.askdirectory()

# plot predictions for images
plot_predicted(imageDirectory, outputDirectory, model, cfg)

