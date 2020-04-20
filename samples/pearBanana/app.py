# 导入Flask类
from flask import Flask, request, render_template, jsonify, redirect
#####################################
# import flask_detect as fd

########################Flask

# 实例化，可视为固定格式
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='')


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
import yellowGreen

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/res101-3class/mask_rcnn_fruit_0062.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/fruit/test")


UPLOAD_FOLDER = os.path.join(ROOT_DIR, "samples/pearBanana/upload_images")
ALLOWED_EXTENSIONS = set(['jpg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

    user_file_names = next(os.walk(UPLOAD_FOLDER))[2]
    names_chosen = random.choice(user_file_names)
    image = skimage.io.imread(os.path.join(UPLOAD_FOLDER, names_chosen))
    print('\n-----------------',len([image]), '---------------\n')
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]  ### the length of this will be the count of items found
    print(r['class_ids'])
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],   #注意我修改了visualize.py 的line166，因为此前安装的MASK RCNN的此文件并没有改动，多以需要再去主文件夹运行python setup.py install
                                class_names, r['scores'])
    print('executed detect_onsite')
    return 'completed detecting: ' + names_chosen
###############################################################
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
###############################################################
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg'))
            return redirect('/detect')
        else:
            print('file type is not correct')
            return render_template('upload.html')


@app.route('/detect')
def detect():
    detect_onsite(model, IMAGE_DIR)
    return render_template('result_detect.html')

@app.route('/splash')
def splash():
    detect_onsite(model, IMAGE_DIR)
    return render_template('result_splash.html')
'''
Main function to run Flask server
'''
if __name__ == '__main__':
    app.run()
