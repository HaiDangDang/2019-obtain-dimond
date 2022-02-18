import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.config import Config
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
# Import COCO config
from masking_objects.ballon_sample_mask import BalloonDataset,BalloonConfig,color_splash
# define the test configuration
config = BalloonConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

dataset = BalloonDataset()
dataset.load_balloon('./tree', "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights('mask_rcnn_balloon_0030.h5', by_name=True)

image = skimage.io.imread('aaa.png')

# image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
%time
results = model.detect([image], verbose=0)


r = results[0]
a = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'])
plt.figure()
plt.imshow(a)
plt.show()
dataset.image_info[0]
image_id =3
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
len(class_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
results = model.detect([image], verbose=1)

r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'])


info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)
splash = color_splash(image, r['masks'])
display_images([splash], cols=1)
# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
mask =  image.astype(np.uint32).copy()
colo = visualize.random_colors(3)
a = visualize.draw_boxes(mask, r['rois'])
plt.imshow(a)
plt.show()
visualize.save_image(image, image_name, r['rois'], r['masks'],
    r['class_ids'],r['scores'],coco_class_names,
    filter_classs_names=['bottle', 'wine glass'],scores_thresh=0.9,mode=0)
# Load random image and mask.
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

a = visualize.save_image(image, 'aaa', r['rois'], r['masks'],
    r['class_ids'],r['scores'],dataset.class_names,
    filter_classs_names=['bottle', 'wine glass'],scores_thresh=0.9,mode=0)
plt.
N = r['rois'].shape[0]
import Image