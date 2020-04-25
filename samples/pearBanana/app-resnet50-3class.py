'''
Some script functions are derived and modified based on the sample tutorials-ballons.py in the repo
Credit should be given to the Repo owner Matterport, Inc

honour the original author:
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
tutorial originally Written by Waleed Abdulla
'''

'''
A simple flask application allow users to select detection or applying color splash on an uploaded images.

The app will load the pretrained model and initialize it during the flask setup progress
It is found that the computer must run the detection immediately loaded the model in order to save the model into the
memory, otherwise users will not be able to run detection at all, as the model is not in the memory

Note:
--Only jpg images will be allowed.
--Depending on the computational power, the initializing time and detection time can vary [greatly].
--It is highly recommend that users should have a decent graphic card, and have NVIDIA GPU Computing Toolkit installed
--This repo is found to be only runnable on a specific combinations of library versions shown as below:
GPU: Acceptable graphic cards, here we using GTX1080
CUDA: V10.0
tensorflow: 1.14-gpu
keras:2.1.3
'''
import os
import sys
import random
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, render_template, jsonify, redirect
from datetime import timedelta

from werkzeug.utils import secure_filename

import mrcnn.model as modellib
from mrcnn import visualize

import fruit3class
###############################configure necessary path#######################################################
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import config
sys.path.append(os.path.join(ROOT_DIR, "samples/pearBanana/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/res50-3class/mask_rcnn_fruit_0075.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "samples/pearBanana/static/initializeImage")


UPLOAD_FOLDER = os.path.join(ROOT_DIR, "samples/pearBanana/upload_images")
ALLOWED_EXTENSIONS = set(['jpg'])

########################configure flask object#################
app = Flask(__name__, template_folder='')
# avoid caching, which prevent showing the detection/splash result
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class InferenceConfig(fruit3class.FruitConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BACKBONE = "resnet50"

    POST_NMS_ROIS_INFERENCE = 2000

    # proved->the higher the image quality, the better the detection accuracy
    # How every, the detection speed will be slowed dramatically
    # depending on your computational power,
    # you might need to modify the 'IMAGE_MAX_DIM = 3520' to fit your graphic card memory capacity
    # as a guidance, we use GTX1080 with 8gb memory, 3520p is the maximum resolution can be dealt with.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 3520 # was 1024

    RPN_NMS_THRESHOLD = 0.7

    # Minimum probability value to accept a detected instance
    DETECTION_MIN_CONFIDENCE = 0.6

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 200

###### create model in inference mode, must run a detection imeediately to save the model to computer memory #######
config = InferenceConfig()

print('\n\n -----Please be patient, the initializing process can take a while depending on your computability-----\n\n')
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'pear', 'banana-ripe', 'banana-nonRipe']

file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
print('\n\n -----Please be patient, almost done initialing-----\n\n')
# Run detection
results = model.detect([image], verbose=1)
r = results[0]  ### the length of this will be the count of items found
print('\n\n -----Initialization Complete -----\n\n')

def detect_onsite(model):
    class_names = ['BG', 'pear', 'banana-ripe', 'banana-nonRipe']

    user_file_names = next(os.walk(UPLOAD_FOLDER))[2]
    names_chosen = random.choice(user_file_names)
    image = skimage.io.imread(os.path.join(UPLOAD_FOLDER, names_chosen))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]  ### the length of this will be the count of items found
    print('the class id of all detected objects as follows')
    print('1: pear, 2: ripe banana, 3: nonripe banana')
    print(r['class_ids'], '\nthere are', len(r['class_ids']), 'fruits detected')

    # Modified visualize.py line166, so need to run 'python setup.py install' again
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    print('executed detect_onsite')
    return 'completed detecting: ' + names_chosen
########################### only accepet jpg file ####################################
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
####################### implement color spalsh effect ########################################
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model):
    # Run model detection and generate the color splash effect
    # Read image
    user_file_names = next(os.walk(UPLOAD_FOLDER))[2]
    names_chosen = random.choice(user_file_names)
    image = skimage.io.imread(os.path.join(UPLOAD_FOLDER, names_chosen))
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = color_splash(image, r['masks'])
    # # Save output
    skimage.io.imsave('static/images/splash_result.jpg', splash)
    print('executed color splash')
################################################################
@app.route('/')
def home():
    if request.method == 'GET':
        return render_template('index.html')

    return render_template('index.html')


@app.route('/UploadDetect', methods=['GET', 'POST'])
def upload_file_detect():
    if request.method == 'GET':
        return render_template('upload_detect.html')

    # only run detect after accepting a valid jpg image
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg'))
            return redirect('/detect')
        else:
            print('file type is not correct')
            return render_template('upload_detect.html')

@app.route('/UploadSplash', methods=['GET', 'POST'])
def upload_file_splash():
    if request.method == 'GET':
        return render_template('upload_splash.html')

    # only run splash after accepting a valid jpg image
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg'))
            return redirect('/splash')
        else:
            print('file type is not correct')
            return render_template('upload_splash.html')


@app.route('/detect')
def detect():
    detect_onsite(model)
    return render_template('result_detect.html')

@app.route('/splash')
def splash():
    detect_and_color_splash(model)
    return render_template('result_splash.html')

'''
Main function to run Flask server
'''
if __name__ == '__main__':
    app.run()
