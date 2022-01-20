import datetime
import json
import logging
import os

import click
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
from torch.nn.functional import softmax
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from src.models.model import DetrPascal

# Logger
log = logging.getLogger(__name__)

# Object classes
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
n_classes = len(classes)


class PredictSet:
    """Helper class to load in images for prediction."""

    def __init__(self, images_from):
        self.images_from = images_from
        self.load_from = []
        self.is_url = False
        if os.path.exists(self.images_from):
            if os.path.isdir(self.images_from):
                self.load_from = [
                    f'{self.images_from}/{filename}'
                    for filename in next(os.walk(self.images_from), (None,
                                                                     None,
                                                                     []))[2]
                ]
            elif os.path.isfile(self.images_from):
                self.load_from.append(self.images_from)
        else:
            try:
                requests.get(self.images_from, stream=True)
                self.is_url = True
                self.load_from.append(self.images_from)
            except:
                pass
        self.loader = self.load_image()

    def __len__(self):
        return len(self.load_from)

    def load_image(self):
        while len(self.load_from):
            if self.is_url:
                try:
                    id = self.load_from.pop(0)
                    image = Image.open(requests.get(
                        id, stream=True).raw).convert('RGB')
                except:
                    log.debug(f'Failed to load image from url: {id}')
                    continue
            else:
                try:
                    id = self.load_from.pop(0)
                    image = Image.open(id).convert('RGB')
                except:
                    log.debug(f'Failed to load image from file: {id}')
                    continue
            yield image, id


@click.command()
@click.argument('load_model_from')
@click.argument('images_from')
@click.option('--predictions_to', default='predictions', required=False)
@click.option('--threshold', default=.5, required=False)
@click.option('--batch_size', default=4, required=False)
@click.option('--draw_boxes', default=True, required=False)
def predict(load_model_from,
            images_from,
            predictions_to='predictions',
            threshold=.5,
            batch_size=4,
            draw_boxes=True):
    """Predicts objects in the provided images.
    
    Arguments:
        load_model_from {str}: path to model weights
        images {str}: path to images for prediction OR url
    Keyword arguments:
        predictions_to {str}: path to folder where results are saved. The path does not have
                              to exist. (default 'predictions')
        threshold {float}: Probability threshold for predictions. (default .5) 
        batch_size {int}: Batch size for prediction. (default 4)
        draw_boxes {bool}: Whether to draw the bounding boxes on the images and save them. (default True)
    """
    assert os.path.exists(load_model_from), "The model path does not exist."

    feature_extractor = DetrFeatureExtractor.from_pretrained(
        "mishig/tiny-detr-mobilenetsv3")

    # TODO: Load in our trained model. Below is a placeholder
    model = DetrPascal.load_from_checkpoint(load_model_from,lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    Images = PredictSet(images_from)
    assert Images, "No images were found."

    # Initialize {predictions_to} folder
    if os.path.exists(predictions_to):
        dirs = next(os.walk(predictions_to))[1]
        dirs_numerated = [int(dir) for dir in dirs]
        next_dir = str(max(dirs_numerated) + 1)
    else:
        os.mkdir(predictions_to)
        next_dir = str(1)
    os.mkdir(f'{predictions_to}/{next_dir}')

    # Save configuration info and results.
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated = {
        'timestamp': ts,
        'model_path': load_model_from,
        'threshold': threshold,
        'results': None
    }
    results = []

    i = 0
    # Prediction
    while len(Images):
        # Construct batch
        batch = []
        batch_results = []
        batch_images_out = []
        while len(batch) < batch_size and len(Images):
            image, id = next(Images.loader)
            if draw_boxes:
                batch_images_out.append(image.copy())
            batch.append(image)
            batch_results.append({'id': i, 'file_name': id, 'detections': []})
            i += 1
        bsize = len(batch) # Actual batch size

        encoding = feature_extractor(images=batch, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**encoding)

        # We loop over each image in the batch
        for logits, bboxes, result in zip(outputs.logits,
                                                     outputs.pred_boxes,
                                                     batch_results):
            probs = softmax(logits, dim=1)
            best = np.argmax(probs, 1)
            criteria = probs[np.arange(len(best)), best] > threshold
            for pred, box, prob in zip(best[criteria], bboxes[criteria],
                                       probs[criteria]):
                # If 'No Object' is detected, skip over the detection
                if pred < n_classes:
                    pred_class = classes[pred]
                else:
                    continue
                
                confidence = prob[pred].item()
                x, y, w, h = tuple(box)
                x, y, w, h = x.item(), y.item(), w.item(), h.item()
                result['detections'].append(
                    (pred_class, (x, y, w, h), confidence)) 
            results.append(result)
        
        if draw_boxes:
            for j, image_out in enumerate(batch_images_out):
                idx = i - bsize + j
                image_w, image_h = image_out.size
                draw = ImageDraw.Draw(image_out, "RGBA")
                for detection in results[idx]['detections']:
                    pred_class, (x, y, w, h), confidence = detection
                    x *= image_w
                    y *= image_h
                    w *= image_w
                    h *= image_h
                    draw.rectangle(
                        (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
                        outline='red',
                        width=2)
                    draw.text((x - w / 2 + 5, y - h / 2 + 5),
                                pred_class,
                                fill='red')
                image_out.save(f'{predictions_to}/{next_dir}/im_pred{idx}.jpg')
    annotated['results'] = results

    # Save info to a readable json file.
    with open(f'{predictions_to}/{next_dir}/annotated.json', 'w') as file:
        json.dump(annotated, file, indent=4)

    return None


if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    predict()
