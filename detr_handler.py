import base64
import io

import torch
from PIL import Image
from torch import device
from transformers import DetrFeatureExtractor
from ts.torch_handler.object_detector import ObjectDetector
from ts.utils.util import map_class_to_label


class DetrHandler(ObjectDetector):
    def initialize(self, context):
        super().initialize(context)

        properties = context.system_properties
        self.threshold = 0
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50")
        self.initialized = False
        self.device = device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True
    
    def preprocess(self, data):
        """
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)
        encoding = self.feature_extractor(images=images,
                                          return_tensors='pt')
        return encoding
    
    
    def inference(self, data):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        pixel_values, pixel_mask = data['pixel_values'].to(self.device), data['pixel_mask'].to(self.device)
        with torch.no_grad():
            results = self.model(pixel_values, pixel_mask)
        return results
    
    
    def postprocess(self, data):
        result = []
        data = [{'boxes': data[0], 'labels': data[1], 'scores': data[2]}]
        box_filters = [(row['scores'] >= self.threshold) & (row['labels'] < (len(self.mapping)-1)) for row in data]
        filtered_boxes, filtered_classes, filtered_scores = [
            [row[key][box_filter].tolist() for row, box_filter in zip(data, box_filters)]
            for key in ['boxes', 'labels', 'scores']
        ]

        for classes, boxes, scores in zip(filtered_classes, filtered_boxes, filtered_scores):
            retval = []
            for _class, _box, _score in zip(classes, boxes, scores):
                _retval = map_class_to_label([[_box]], self.mapping, [[_class]])[0]
                _retval['score'] = _score
                retval.append(_retval)
            result.append(retval)

        return result
