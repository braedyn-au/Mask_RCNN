"""
Mask R-CNN
Train on septin masks

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 Septin.py train --dataset=/path/to/Septin/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 Septin.py train --dataset=/path/to/Septin/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 Septin.py train --dataset=/path/to/Septin/dataset --weights=imagenet

    # Apply color splash to an image
    python3 Septin.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 Septin.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import listdir

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SeptinConfig(Config):
    """Configuration for training on the septin dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Septin_cfg"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Septin

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 233

    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.5

    # Decrease learning momentum
    LEARNING_MOMENTUM = 0.5

    # BATCH_SIZE = 2


############################################################
#  Dataset
############################################################

class SeptinDataset(Dataset):

    def load_Septin(self, subset):
        """Load a subset of the Septin dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or test
        """
        # Add classes. We have only one class to add.
        self.add_class("Septin", 1, "Septin")

        # Labels are in same folder as images
        assert subset in ["train", "test"]
        dataset_dir =  './Images/'
        dataset_dir = os.path.join(dataset_dir, subset)
        
        for img in listdir(dataset_dir):
            if img.endswith('.tif'):
                label = img[:-4]+'.json'
                if os.path.exists(os.path.join(dataset_dir,label)):
                    # Load annotations
                    annotations = json.load(open(os.path.join(dataset_dir, label)))
                    circles = []
                    polygons = []
                    regions = [a for a in annotations['shapes']]
                    # Take the circle coordinates
                    for a in regions:
                        if a['shape_type'] == 'circle':
                            c,r = a['points'][0]
                            c2,r2 = a['points'][1]
                            radius = np.sqrt((c2-c)*(c2-c)+(r2-r)*(r2-r))
                            circles.append((int(r),int(c),int(radius)))
                        if a['shape_type'] == 'polygon':
                            rpoints = []
                            cpoints = []
                            for p in a['points']:
                                rpoints.append(p[1])
                                cpoints.append(p[0])
                            polygons.append([rpoints,cpoints])
                else:
                    # Empty image as a true negative
                    circles = []
                    polygons = []

                image_path = os.path.join(dataset_dir, annotations['imagePath'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "Septin",
                    image_id=annotations['imagePath'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    circles=circles, polygons=polygons)
            else:
                pass

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Septin dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Septin":
            return super(self.__class__, self).load_mask(image_id)

        # Convert circles to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        maskTemp = np.zeros([info["height"]+200, info["width"]+200, len(info["circles"])+len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["circles"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.circle(p[0]+100, p[1]+100, p[2])
            maskTemp[rr, cc, i] = 1
        for i, p in enumerate(info["polygons"]):
            dimension = i + len(info["circles"])
            rrp, ccp = skimage.draw.polygon(np.add(p[0],100), np.add(p[1],100))
            maskTemp[rrp,ccp, dimension] = 1
        # Crop to include edges
        mask = maskTemp[100:-100,100:-100]

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Septin":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(config=SeptinConfig()):
    """Train the model."""
    # Training dataset.
    dataset_train = SeptinDataset()
    dataset_train.load_Septin("train")
    dataset_train.prepare()
    print('Train: %d' % len(dataset_train.image_ids))

    # Validation dataset
    dataset_val = SeptinDataset()
    dataset_val.load_Septin("test")
    dataset_val.prepare()
    print('Test: %d' % len(dataset_val.image_ids))

    config = SeptinConfig()
    config.display()

    # define the model
    model = MaskRCNN(mode='training', model_dir='./', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=8,
                layers='heads')


############################################################
#  Training
############################################################

train()
