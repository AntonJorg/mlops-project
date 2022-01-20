import logging
import os

import click
import torch
import numpy as np

from collections import namedtuple
from model import DetrPascal
from torch.nn.functional import softmax
from PIL import Image
from torchvision import transforms
from transformers import DetrFeatureExtractor


class DetrPascalSerializable(DetrPascal):

    def __init__(self, model, lr, lr_backbone, weight_decay):
        super().__init__(lr, lr_backbone, weight_decay)
        self.model = model

    def forward(self, pixel_values, pixel_mask):
        Output = namedtuple('Out', ['boxes', 'labels', 'scores'])
        model_outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        probs = softmax(model_outputs.logits.squeeze(0), dim=1)
        boxes = model_outputs.pred_boxes.squeeze(0)
        x, y, w, h = boxes[:, 0].unsqueeze(-1), boxes[:, 1].unsqueeze(
            -1), boxes[:, 2].unsqueeze(-1), boxes[:, 3].unsqueeze(-1)
        boxes = torch.cat((x - w/2, y - h/2, x + w/2, y + h/2), 1)
        labels = torch.argmax(probs, 1)
        scores = probs[(labels > -1), labels]
       
        output = Output(boxes, labels, scores)
        return output


# Logger
log = logging.getLogger(__name__)


@click.command()
@click.argument('load_model_from', type=click.Path(exists=True))
@click.option('--serialized_to',
              default='models/serialized_models',
              required=False)
def serialize(load_model_from, serialized_to='models/serialized_models'):
    model = DetrPascal.load_from_checkpoint(load_model_from,
                                            lr=1e-4,
                                            lr_backbone=1e-5,
                                            weight_decay=1e-4
                                            )
    model = DetrPascalSerializable(model,
                                   lr=1e-4,
                                   lr_backbone=1e-5,
                                   weight_decay=1e-4)
    
    log.info(f'Succesfully loaded model from {load_model_from}')
    model.eval()
    model_name = os.path.basename(load_model_from)[:-5]
    
    if not os.path.exists(serialized_to):
        os.mkdir(serialized_to)

    save_path = f'{serialized_to}/{model_name}.pt'
    
    feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50")
    example_input = torch.rand(400, 400, 3) * 255
    encoding = feature_extractor(images=[example_input],
                                 return_tensors='pt')
    print(model(**encoding))
    traced_model = torch.jit.trace(model, [encoding['pixel_values'], encoding['pixel_mask']])

    torch.jit.save(traced_model, save_path)
    log.info(f'Succesfully saved the serialized model at {save_path}')


if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    serialize()
