
# Custom Object Detection and Novel Bounding Box Metric with YOLO

This project focuses on building a custom YOLOv5 model to detect cats and dogs in images. It walks through preparing the data, processing annotations, training the model, and assessing its performance.

## Installation

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).


```bash
git clone --branch main https://github.com/SasiniWanigathunga/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## Data

The dataset used in this project consists of images and annotations for detecting cats and dogs. You'll need to specify the paths for the annotations, images, and output directories when running the preprocessing script.

### 1. Preprocess Data

Download the dataset from [**here**](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detectionThe) and place the training and validations samples in the following structure.

```bash
.
├── data
│   ├── train_sample_dataset
│   │   ├── images
│   │   │   ├── train          # Training images
│   │   │   ├── val            # Validation images
│   │   ├── labels
│   │   │   ├── train          # Corresponding labels for training images (.xml format)
│   │   │   ├── val            # Corresponding labels for validation images (.xml format)
│   ├── test_images          # Images for testing/inference
│   ├── hyp.yolo_voc.yaml    # Hyperparameter configuration file
│   ├── yolo_voc.yaml        # YOLO dataset configuration file
```

### 2. Training

Run the following command to start training the YOLOv5 model:

```bash
python train.py  --weights yolov5xu.pt --data data/yolo_voc.yaml --cfg models/yolov5s_yolo_voc.yaml --hyp data/hyp.yolo_voc.yaml --epochs 10 --batch-size 10 --custom_loss
```

This will save the model in results/ directory. For the training with custom loss have the custom_loss argument in the above script and otherwise not.

## Model Evaluation

Evaluate the model on the validation set using the following command:

```bash
python evaluate.py  --data data/yolo_voc.yaml --weights result/epoch6_2025-02-16_07-03-49_model.pth --batch 32
```