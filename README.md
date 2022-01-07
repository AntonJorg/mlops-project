# mlops-project
Project work for 02476 Machine Learning Operations

## Goal
We plan to use the huggingface transformer framework to train a DETR model for object detection on the Pascal VOC 2012 Dataset. We wish to make our results reproducible and make the simplest and most readable implementation possible.

## Framework
The tranformer framework supports the DETR object detection model and pretrained weights can easily be applied. We plan to use the pretraining feature as well as the DETR model classes to obtain a model which can readily be finetuned on a new dataset. The main motivation for using the transformer framework is to take care of the hard parts of the implementation, such that we do not have to think about model architecture and efficiency directly.

## Model
DETR is an end-to-end trainable tranformer neural network for object detection. It solves some of the issues of previous methods, mainly by being simpler to implement.

## Data
The Pascal VOC 2012 Dataset is an object detection dataset with 17112 images and 20 classes and is commonly used as a benchmark for image detection models.
