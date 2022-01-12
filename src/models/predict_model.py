import os
import click
import requests

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.functional import softmax
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrFeatureExtractor

# Object classes
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person","pottedplant", "sheep", "sofa", "train", "tvmonitor"]
n_classes = len(classes)

@click.command()
@click.argument('load_model_from', type=click.Path(exists=True))
@click.argument('images_from')
@click.option('--predictions_to', default='predictions', required=False)
@click.option('--threshold', default=.5, required=False)
def predict(load_model_from, images_from, predictions_to='predictions', threshold=.5):
    """Predicts objects in the provided images.
    
    Arguments:
        load_model_from {str}: path to model weights
        images {str}: path to images for prediction OR url
    """
    assert os.path.exists(load_model_from), "The model path does not exist."
    
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    # TODO: Load in our trained model. Below is a placeholder
    model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50", num_labels=n_classes, ignore_mismatched_sizes=True
                )

    images = []
    # Load in images
    if os.path.exists(images_from):
        if os.path.isdir(images_from):
            filenames = next(os.walk(images_from), (None, None, []))[2]
            for f in filenames:
                try:
                    image = Image.open(f).convert('RGB')
                    images.append(image)
                except:
                    continue
                
        elif os.path.isfile(images_from):
            try:
                image = Image.open(images_from).convert('RGB')
                images.append(image)
            except:
                pass
    else:
        try:
            image = Image.open(requests.get(images_from, stream=True).raw).convert('RGB')
            images.append(image)
        except:
            pass
    
    assert len(images), "No images were found."

    images_out = []
    # We predict the images 1 by 1, ie. no batching.
    for image in images:
        image_w, image_h = image.size
        encoding = feature_extractor(images=image, return_tensors='pt')

        with torch.no_grad():
            output = model(**encoding)
        
        image_out = image.copy()
        draw = ImageDraw.Draw(image_out, "RGBA")
        # Here we don't actually need the for loop,
        # but for batches, we would loop over each image in the batch
        for logits, bboxes in zip(output.logits, output.pred_boxes):
            probs = softmax(logits, dim=1)
            best = np.argmax(probs, 1)
            criteria = probs[np.arange(len(best)), best] > threshold
            # preds = best[criteria]
            for pred, box in zip(best[criteria], bboxes[criteria]):
                pred_class = classes[pred] if pred < n_classes else "No Object"
                x,y,w,h = tuple(box)
                x *= image_w
                y *= image_h
                w *= image_w
                h *= image_h
                draw.rectangle((x - w/2, y - h/2, 
                                x + w/2, y + h/2), outline='red', width=2)
                draw.text((x - w/2 + 5, y - h/2 + 5), pred_class, fill='red')
        images_out.append(image_out)

        
        
        if os.path.exists(predictions_to):
            dirs = next(os.walk(predictions_to))[1]
            dirs_numerated = [int(dir) for dir in dirs]
            next_dir = str(max(dirs_numerated) + 1)
        else:
            os.mkdir(predictions_to)
            next_dir = str(1)
        os.mkdir(f'{predictions_to}/{next_dir}')
        
        for i, image_out in enumerate(images_out):
            image_out.save(f'{predictions_to}/{next_dir}/im_pred{i+1}.jpg')
    
    return None

if __name__ == '__main__':
    predict()
