# Mask R-CNN for Object Detection and Segmentation

This is an [implementation](https://github.com/matterport/Mask_RCNN) of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras,TensorFlow and Flask. 

The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101/ResNet50 backbone. The flask provides a simple RESTful API to users, so users can run the  detection on pre-trained model easily via a simple website. 

![Instance Segmentation Sample](assets/sample-3class-takes6min30secs-178detection-allCorrect.jpg)

## Honour Original Work
The implementation of the MASK RCNN framework is written by Matterport , his work is honoured. 
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

## Requirements
Python 3.6.8, TensorFlow 1.14 - gpu, Keras 2.1.3.

Other common packages listed in `requirements.txt`.


## Step-by-step Installation Guide
1. Clone this repository to your desired local project directory

    ```
    YOUR_PROJECT_DIR
    └── Mask_RCNN
    ```

2. If you have a Nvidia GPU, it is **compulsory** to install a GPU version of tensorflow in order to run this repo. To install, issue the command shown below. Please also follow step 3 and 4 to setup your graphic card. 

    ```bash
      pip install tensorflow-gpu==1.14
    ```

   If you don't have a Graphic Card, simply install tensorflow via

    ```bash
      pip install tensorflow==1.14
    ```

3. **[Follow only If you have a GPU card]**Go to the [Nvidia Website](https://developer.nvidia.com/cuda-10.0-download-archive), Follow the instructions to install CUDA Toolkit 10.0 Archive. 

   1. Note that you may need to install Microsoft Visual Studio 2017 if the installation program asks.
   
4. **[Follow only If you have a GPU card]**Download [cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.0](https://developer.nvidia.com/rdp/cudnn-download#a-collapse765-10)] zip file and unzip it to a folder. 

    1. Copy all the cuDNN components to 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0'

    2. Copy the unziped cuda folder to C:\tools, so that cuda exists at C:\tools\cuda

    3. **[IMPORTANT]** Setup path of Toolkit, otherwise tensorflow-gpu will not be able to run. 

       Issue the following command in the terminal:

       ```
       SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%PATH%
       SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64;%PATH%
       SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;%PATH%
       SET PATH=C:\tools\cuda\bin;%PATH%
       ```

5. Restart the computer. Test you have your GPU Toolkit ready for tensorflow. If you encountered any error, please reinstall again. Here are some useful Tutorials to help to install the Tensorflow, CUDA and cuDNN. 

   1. [[Useful Configuration Tutorial]](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)  

      [[Tutorials1]](https://www.tensorflow.org/install/gpu) 

      [[Tutorial2]](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows) 

   2. Issue the command 'nvcc -V' to verify CUDA and cuDNN installation, you should see the version of your cuda tools. 

      ![Instance Segmentation Sample](assets/cuda_test.jpg)

   3. Test You can start the tesnforflow-GPU without errors by issue the command in python 

       ```bash
       import tensorflow as tf
       hello = tf.constant('Hello, TensorFlow!')
       sess = tf.Session()
       print(sess.run(hello))
       ```
   ![Instance Segmentation Sample](assets/gpu_test.jpg)

6. Then install Keras v2.1.3 via pip command

   ```bash
   pip install keras==2.1.3
   ```

7. Navigate to the Mask RCNN folder, Install dependencies

   ```bash
   pip3 install -r requirements.txt
   ```

8. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ```
    
9. Download the pretrained model from [xxxxxxxxxxx].

    1. Create a folder named logs under the Mask_RCNN root directory
    2. Unzip the downloaded models to 'Mask_RCNN/logs'
    3. Resultantly, your project directory should looks like below:

```
Mask_RCNN
    ├── assets        
    ├── logs							
    |   └── res50-3class  
    |   |   └── mask_rcnn_fruit_0075.h5 //pre-trained model 
    |   └── res101-2class  
    |   |   └── mask_rcnn_fruit_0065.h5 //pre-trained model
    |   └── res101-3class  
    |   |   └── mask_rcnn_fruit_0062.h5 //pre-trained model
    ├── mrcnn     //The Mask_RCNN framework
    └── samples
        └── pearBanana
    |    |   |   └── static
    |    |   | 		└──images			//folders to store the detection result
    |    |   | 		|	└──detection_result.jpg
    |    |   | 		|	└──splash_result.jpg
       			    └──initializeImag	//contain a image to initialize the model
    |    |   |   └── upload_images		//folder to store the uploaded image by users
    |    |   |   └──app-2class.py		//server 1 to start detection on 2 classes 
    |    |   |   └──app-3class.py		//server 2 to start detection on 3 classes
    |    |   |   └──fruit.py			//training script
    |    |   |   └──fruit3class.py		//training script
    |    |   |   └──5 htmls pages		//webpage used for Flask Server
    
```