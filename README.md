# YOK: Yolov2 On Keras

## Features
- Object detection in a image/video using yolov2 based on keras framework(tensorflow backend).
- It includes training and inference.
- Yolov2 and TinyYolov2 models can be trained.
- It is tested using COCO dataset.

## Requirements
* Keras
* tensorflow
* scipy
* matplotlib
* numpy
* tqdm
* opencv-python
* baker
* cytoolz
* lxml
* pandas
* pillow
* imgaug

Note: Please take note for tensorflow gpu support, install the cuda driver, cuda toolkit and cudnn
from  this [link] (https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) and the
corresponding tensorflow-gpu version supported by refering to this [link](https://www.tensorflow.org/install/source#tested_build_configurations)

## Inference/Prediction
- Please download the pre-trained models from this [link](https://drive.google.com/open?id=1fcqa1-mzvgKSADBtTdJGHxWVfDmWlDkO)
- Alternatively one can re-train the model files as described in the Training section
- Move the weights file under `h5_models/pre-trained`
- Adjust model to use Yolov2/TinyYolov2, pre-trained weights file path and the class to target/detect in `config_predict.json`
- Run the script `PYTHONPATH=lib:$PYTHONPATH CUDA_VISIBLE_DEVICES="" python3 predict.py` (You can adjust the GPU devices to be used using CUDA_VISIBLE_DEVICES)
- By defualt the video under videos/conference.mp4 is used as input and the output video is saved as videos/conference_detect.avi and "class=person" is used for detection
- The csv file contatining the number of person per second of the video is saved as logs/number_count.csv
- One can input an image file too using command `PYTHONPATH=lib:$PYTHONPATH CUDA_VISIBLE_DEVICES="" python3 predict.py -i images/person.jpg -o output_images/person.jpg`
- Run `PYTHONPATH=lib:$PYTHONPATH python3 predict.py --help` to see script arguments and defaults.

### Output
![Inference Output](meta/conference_detect.gif)

## Training
- For training, pre-trained weights files (h5 models) are used.
- Please download the pre-trained models from this [link](https://drive.google.com/open?id=1fcqa1-mzvgKSADBtTdJGHxWVfDmWlDkO)
- Move the weights file under `h5_models/pre-trained`
- Download the COCO dataset
  You can create annotations using the two following ways:
  - Convert COCO annotations to VOC format:
    - Train set : http://images.cocodataset.org/zips/train2014.zip
    - Validation set: http://images.cocodataset.org/zips/val2014.zip
    - Annotations : http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    - Extract train2014.zip and val2014.zip and move the train2014 and val2014 directory under `data/coco/images`
    - Extract annotations_trainval2014.zip and move the annotations_trainval2014 directory under `data/coco`
    - Go to data directory and run the script `python3 coco2pascal.py`
    - The annotations will be saved under `data/coco/pascal_format` (It will take considerable amount of time)
  - Download the VOC annotations for COCO dataset:
    - Download the annotations from this [link](https://drive.google.com/open?id=1V-w65XowVcQHf4xEBH6hoSLEb-Hr58-h)
    - Unzip pascal_format.zip and move the directory pascal_format under `data/coco`
- Adjust epochs, batch_size in `config_train.json`, also the pre-trained weights file path and which model to use Yolov2 or TinyYolov2
- Also specify the train and eval data path in `config_train.json`
- Run the script `PYTHONPATH=lib:$PYTHONPATH CUDA_VISIBLE_DEVICES="" python3 train.py` (You can adjust the GPU devices to be used using CUDA_VISIBLE_DEVICES)
- The output model file is saved in `h5_models/trained`
- Run `PYTHONPATH=lib:$PYTHONPATH python3 train.py --help` to see script arguments and defaults.
