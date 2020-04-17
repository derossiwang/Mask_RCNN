# 导入Flask类
from flask import Flask, request, render_template, jsonify
#####################################
# import flask_detect as fd

########################Flask

# 实例化，可视为固定格式
app = Flask(__name__, template_folder='')


###################################
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import config
sys.path.append(os.path.join(ROOT_DIR, "samples/pearBanana/"))  # To find local version
import yellowGreen

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/res101-3class/mask_rcnn_fruit_0062.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/fruit/test")

class InferenceConfig(yellowGreen.FruitConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    POST_NMS_ROIS_INFERENCE = 2000

    # proved->the higher the image quality, the better the detection accuracy will be increased
    # How every, the detection speed will be slowed dramatically
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 3520 # was 1024

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.6

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 200

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

#################测试后发现必须得在全局先运行一次detection才可以!!!!!!!!!!!!
class_names = ['BG', 'pear', 'banana-ripe', 'banana-nonRipe']

file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)
r = results[0]  ### the length of this will be the count of items found
print(r['class_ids'],'\ncompleted initializing\n')

def detect_onsite(model, IMAGE_DIR):
    # class_names = ['BG', 'banana', 'pear']
    class_names = ['BG', 'pear', 'banana-ripe', 'banana-nonRipe']

    user_file_names = next(os.walk(IMAGE_DIR))[2]
    names_chosen = random.choice(user_file_names)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, names_chosen))
    print('\n-----------------',len([image]), '---------------\n')
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]  ### the length of this will be the count of items found
    print(r['class_ids'])
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])
    print('executed detect_onsite')
    return 'completed detecting: ' + names_chosen
###############################################################

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return detect_onsite(model, IMAGE_DIR)

'''
Main function to run Flask server
'''
if __name__ == '__main__':
    app.run()
