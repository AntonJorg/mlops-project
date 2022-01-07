from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import json
import requests
import os
import torch

batch_size=4

path = os.path.join(os.getcwd(), 'data', 'train')
file = open(os.path.join(path, 'new_annotations.json'), 'r')
annotations=json.load(file)
images = [Image.open(annotations[i]['image_location']) for i in range(batch_size)]
annotations_t = [annotations[i] for i in range(batch_size)]

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
inputs = feature_extractor(images=images, annotations=annotations_t, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes





