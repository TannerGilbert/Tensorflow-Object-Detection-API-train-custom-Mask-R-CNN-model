# Train a Mask R-CNN model with the Tensorflow Object Detection API

[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)

![Mask R-CNN prediction](doc/prediction_3.png)

## 1. Installation

You can install the TensorFlow Object Detection API either with Python Package Installer (pip) or Docker, an open-source platform for deploying and managing containerized applications. For running the Tensorflow Object Detection API locally, Docker is recommended. If you aren't familiar with Docker though, it might be easier to install it using pip.

First clone the master branch of the Tensorflow Models repository:

```bash
git clone https://github.com/tensorflow/models.git
```

### Docker Installation

```bash
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf1/Dockerfile -t od .
docker run -it od
```

### Python Package Installation

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
python -m pip install .
```

> Note: The *.proto designating all files does not work protobuf version 3.5 and higher. If you are using version 3.5, you have to go through each file individually. To make this easier, I created a python script that loops through a directory and converts all proto files one at a time.

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

```
python use_protobuf.py <path to directory> <path to protoc file>
```

To test the installation run:

```bash
# Test the installation.
python object_detection/builders/model_builder_tf1_test.py
```

If everything installed correctly you should see something like:

```bash
Running tests under Python 3.6.9: /usr/bin/python3
[ RUN      ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(True)
[       OK ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(True)
[ RUN      ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(False)
[       OK ] ModelBuilderTF1Test.test_create_context_rcnn_from_config_with_params(False)
[ RUN      ] ModelBuilderTF1Test.test_create_experimental_model
[       OK ] ModelBuilderTF1Test.test_create_experimental_model
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(True)
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(True)
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(False)
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_from_config_with_crop_feature(False)
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_model_from_config_with_example_miner
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_model_from_config_with_example_miner
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[       OK ] ModelBuilderTF1Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[ RUN      ] ModelBuilderTF1Test.test_create_rfcn_model_from_config
[       OK ] ModelBuilderTF1Test.test_create_rfcn_model_from_config
[ RUN      ] ModelBuilderTF1Test.test_create_ssd_fpn_model_from_config
[       OK ] ModelBuilderTF1Test.test_create_ssd_fpn_model_from_config
[ RUN      ] ModelBuilderTF1Test.test_create_ssd_models_from_config
[       OK ] ModelBuilderTF1Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF1Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF1Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF1Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF1Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF1Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF1Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF1Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF1Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF1Test.test_session
[  SKIPPED ] ModelBuilderTF1Test.test_session
[ RUN      ] ModelBuilderTF1Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF1Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF1Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF1Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF1Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF1Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 21 tests in 0.163s

OK (skipped=1)
```

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