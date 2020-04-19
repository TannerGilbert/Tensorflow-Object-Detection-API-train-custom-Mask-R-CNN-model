# Train a Mask R-CNN model with the Tensorflow Object Detection API

![Mask R-CNN prediction](doc/prediction_3.png)

## 1. Installation

### Clone the repository and install dependencies

First, we need to clone the Tensorflow models repository. This can be done by either cloning the repository directly or by typing **git clone https://github.com/tensorflow/models --single-branch --branch r1.13.0** inside a command line.

After cloning the repository, it is a good idea to install all the dependencies. This can be done by typing:

```bash
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

Also make sure to use Tensorflow 1.x since training a custom model doesn't work with Tensorflow 2 yet.

### Install the COCO API

COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. If you want to use the data-set and evaluation metrics, you need to clone the cocoapi repository and copy the pycocotools subfolder to the tensorflow/models/research directory.

```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```

Using make won't work on windows. To install the cocoapi on windows the following command can be used:

```bash
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

### Protobuf Installation/Compilation

The Tensorflow Object Detection API uses .proto files. These files need to be compiled into .py files in order for the Object Detection API to work properly. Google provides a programmed called Protobuf that can compile these files.

Protobuf can be downloaded from this website. After downloading, you can extract the folder in a directory of your choice.

After extracting the folder, you need to go into models/research and use protobuf to extract python files from the proto files in the object_detection/protos directory.

The official installation guide uses protobuf like:

```bash
./bin/protoc object_detection/protos/*.proto --python_out=. 
```

But the * which stands for all files didn’t work for me, so I wrote a little Python script to execute the command for each .proto file.

```python
import os
import sys
args = sys.argv
directory = args[1]
protoc_path = args[2]
for file in os.listdir(directory):
    if file.endswith(".proto"):
        os.system(protoc_path+" "+directory+"/"+file+" --python_out=.")
```

This file needs to be saved inside the research folder, and I named it use_protobuf.py. Now we can use it by going into the console and typing:

```bash
python use_protobuf.py <path to directory> <path to protoc file>  Example: python use_protobuf.py object_detection/protos C:/Users/Gilbert/Downloads/bin/protoc 
```

### Add necessary environment variables and finish Tensorflow Object Detection API installation

Lastly, we need to add the research and research/slim folder to our environment variables and run the setup.py file.

To add the paths to environment variables in Linux you need to type:

```bash
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/object_detection
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/slim
```

On windows you need to at the path of the research folder and the research/slim to your PYTHONPATH environment variable (See Environment Setup).

To run the setup.py file, we need to navigate to tensorflow/models/research and run:

```bash
# From within TensorFlow/models/research/
python setup.py build
python setup.py install
```

This completes the installation of the object detection api. To test if everything is working correctly, run the object_detection_tutorial.ipynb notebook from the object_detection folder.

If your installation works correctly, you should see the following output:

![Tensorflow Object Detection API Tutorial Output](doc/tutorial_output.png)

### Run the Tensorflow Object Detection API with Docker

Installing the Tensorflow Object Detection API can be hard because there are lots of errors that can occur depending on your operating system. Docker makes it easy to setup the Tensorflow Object Detection API because you only need to download the files inside the [docker folder](docker/) and run **docker-compose up**. 

After running the command docker should automatically download and install everything needed for the Tensorflow Object Detection API and open Jupyter on port 8888. If you also want to have access to the bash for training models, you can simply say **docker exec -it CONTAINER_ID**. For more information, check out [Dockers' documentation](https://docs.docker.com/).

If you experience any problems with the docker files, be sure to let me know.

### 2. Gathering data

Now that the Tensorflow Object Detection API is ready to go, we need to gather the images needed for training. 

To train a robust model, we need lots of pictures that should vary as much as possible from each other. That means that they should have different lighting conditions, different backgrounds, and lots of random objects in them.

You can either take the pictures yourself, or you can download pictures from the internet. For my microcontroller detector, I have four different objects I want to detect (Arduino Nano, ESP8266, Raspberry Pi 3, Heltect ESP32 Lora).

I took about 25 pictures of each individual microcontroller and 25 pictures containing multiple microcontrollers using my smartphone. After taking the pictures, make sure to transform them to a resolution suitable for training (I used 800x600).

![](doc/image_gallery.png)

You can use the [resize_images script](resize_images.py) to resize the image to the wanted resolutions.

```bash
python resize_images.py -d images/ -s 800 600
```

After you have all the images, move about 80% to the object_detection/images/train directory and the other 20% to the object_detection/images/test directory. Make sure that the images in both directories have a good variety of classes.

## 3. Labeling data

After you have gathered enough images, it's time to label them, so your model knows what to learn. In order to label the data, you will need to use some kind of labeling software.

For object detection, we used [LabelImg](https://github.com/tzutalin/labelImg), an excellent image annotation tool supporting both PascalVOC and Yolo format. For Image Segmentation / Instance Segmentation there are multiple great annotations tools available. Including, [VGG Image Annotation Tool](http://www.robots.ox.ac.uk/~vgg/software/via/), [labelme](https://github.com/wkentaro/labelme), and [PixelAnnotationTool](https://github.com/abreheret/PixelAnnotationTool). I chose labelme, because of its simplicity to both install and use.

![](doc/labelme_example.jpg)

## 4. Generating Training data

With the images labeled, we need to create TFRecords that can be served as input data for the training of the model. Before we create the TFRecord files, we'll convert the labelme labels into COCO format. This can be done with the [labelme2coco.py script](images/labelme2coco.py).

```bash
python labelme2coco.py train train.json
python labelme2coco.py test test.json
```

Now we can create the TFRecord files using the [create_coco_tf_record.py script](create_coco_tf_record.py).

```bash
python create_coco_tf_record.py --logtostderr --train_image_dir=images/train --test_image_dir=images/test --train_annotations_file=images/train.json --test_annotations_file=images/test.json --output_dir=./
```

After executing this command, you should have a train.record and test.record file inside your object detection folder.

## 5. Getting ready for training

The last thing we need to do before training is to create a label map and a training configuration file.

### 5.1 Creating a label map

The label map maps an id to a name. We will put it in a folder called training, which is located in the object_detection directory. The labelmap for my detector can be seen below.

```bash
item {
    id: 1
    name: 'Arduino'
}
item {
    id: 2
    name: 'ESP8266'
}
item {
    id: 3
    name: 'Heltec'
}
item {
    id: 4
    name: 'Raspberry'
}
```

The id number of each item should match the ids inside the train.json and test.json files.

```json
"categories": [
    {
        "supercategory": "Arduino",
        "id": 0,
        "name": "Arduino"
    },
    {
        "supercategory": "ESP8266",
        "id": 1,
        "name": "ESP8266"
    },
    {
        "supercategory": "Heltec",
        "id": 2,
        "name": "Heltec"
    },
    {
        "supercategory": "Raspberry",
        "id": 3,
        "name": "Raspberry"
    }
],
```

### 5.2 Creating the training configuration

Lastly, we need to create a training configuration file. The Tensorflow Object Detection API provides 4 model options:

From the [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md):
| Model name  | Speed (ms) | COCO mAP[^1] | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [mask_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz) | 771 | 36 | Masks |
| [mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 79 | 25 | Masks |
| [mask_rcnn_resnet101_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz) | 470 | 33 | Masks |
| [mask_rcnn_resnet50_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz) | 343 | 29 | Masks |

For this tutorial I chose to use the mask_rcnn_inception_v2_coco, because it's alot faster than the other options. You can find the [mask_rcnn_inception_v2_coco.config file](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/mask_rcnn_inception_v2_coco.config) inside the samples/config folder. Copy the config file to the training directory. Then open it with a text editor and make the following changes:

* Line 10: change the number of classes to number of objects you want to detect (4 in my case)

* Line 126: change fine_tune_checkpoint to the path of the model.ckpt file:

    * ```fine_tune_checkpoint: "<path>/models/research/object_detection/training/mask_rcnn_inception_v2_coco_2018_01_28/model.ckpt"```

* Line 142: change input_path to the path of the train.records file:

    * ```input_path: "<path>/models/research/object_detection/train.record"```

* Line 158: change input_path to the path of the test.records file:

    * ```input_path: "<path>/models/research/object_detection/test.record"```

* Line 144 and 160: change label_map_path to the path of the label map:

    * ```label_map_path: "<path>/models/research/object_detection/training/labelmap.pbtxt"```

* Line 150: change num_example to the number of images in your test folder.

## 6. Training the model

To train the model execute the following command in the command line:

```bash
python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/mask_rcnn_inception_v2_coco.config
```

Every few minutes, the current loss gets logged to Tensorboard. Open Tensorboard by opening a second command line, navigating to the object_detection folder and typing:

```bash
tensorboard --logdir=training
```

The training script saves checkpoints every few minutes. Train the model until it reaches a satisfying loss, then you can terminate the training process by pressing Ctrl+C.

### Training in Google Colab

If your computer doesn't have a good enough GPU to train the model locally, you can train it on Google Colab. For this, I recommend creating a folder that has the data as well as all the config files in it and putting it on Google Drive. That way, you can then load in all the custom files into Google Colab.

You can find an example inside the [Tensorflow_Object_Detection_API_Instance_Segmentation_in_Google_Colab.ipynb notebook](Tensorflow_Object_Detection_API_Instance_Segmentation_in_Google_Colab.ipynb).

## 7. Exporting the inference graph

Now that we have a trained model, we need to generate an inference graph, which can be used to run the model. For doing so we need to first of find out the highest saved step number. For this, we need to navigate to the training directory and look for the model.ckpt file with the biggest index.

Then we can create the inference graph by typing the following command in the command line.

```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/mask_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

XXXX represents the highest number.