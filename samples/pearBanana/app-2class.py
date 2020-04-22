


from flask import Flask, request, render_template, jsonify, redirect
from datetime import timedelta


from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)  # avoid caching, which prevent showing the detection/splash result
###################################
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')
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
import fruit

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/res101-2class/mask_rcnn_fruit_0065.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "samples/pearBanana/static/initializeImage")


UPLOAD_FOLDER = os.path.join(ROOT_DIR, "samples/pearBanana/upload_images")
ALLOWED_EXTENSIONS = set(['jpg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class InferenceConfig(fruit.FruitConfig):
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
# config.display()
print('\n\n -----Please be patient, the initializing process can take a while depending on your computability-----\n\n')
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

################# Must run the detction once to save the model to memory! #######
class_names = ['BG', 'banana', 'pear']

file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
print('\n\n -----Please be patient, almost done initialing-----\n\n')
# Run detection
results = model.detect([image], verbose=1)
r = results[0]  ### the length of this will be the count of items found
print('\n\n -----Initialization Complete -----\n\n')

def detect_onsite(model):
    class_names = ['BG', 'banana', 'pear']

    user_file_names = next(os.walk(UPLOAD_FOLDER))[2]
    names_chosen = random.choice(user_file_names)
    image = skimage.io.imread(os.path.join(UPLOAD_FOLDER, names_chosen))
    print('\n-----------------',len([image]), '---------------\n')
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]  ### the length of this will be the count of items found
    print(r['class_ids'])
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],   #Modified visualize.py 的line166, so need to run 'python setup.py install' again
                                class_names, r['scores'])
    print('executed detect_onsite')
    return 'completed detecting: ' + names_chosen
###############################################################
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
###############################################################
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
