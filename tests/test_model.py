import torch
from hydra import compose, initialize
from src.models.model import DetrPascal
from transformers import DetrFeatureExtractor

seed = 1510
torch.manual_seed(seed)
batch_size = 4
sizes = [(w, h, 3) for w in (400, 2000)
         for h in (400, 2000)]
images = [torch.rand(size) * 255 for size in sizes]

# Load config file
config_path = '../src/models/config'
try:
    initialize(config_path=config_path, job_name="test_app")
    config = compose(config_name="config.yaml", return_hydra_config=True)
except Exception as err:
    print('An exception occured when loading config.')
    raise

feature_extractor = DetrFeatureExtractor.from_pretrained(
    "facebook/detr-resnet-50")

hparams = config.experiment

model = DetrPascal(lr=hparams.lr,
                   lr_backbone=hparams.lr_backbone,
                   weight_decay=hparams.weight_decay)

# Forward pass
idx = 0
while len(images):
    batch = []
    targets = []
    batch_idx = []
    while len(batch) < batch_size and len(images):
        batch.append(images.pop(0))
        batch_idx.append(idx)
        annotations = []
        num_objects = torch.randint(5, (1, ))
        im_w, im_h, _ = sizes.pop(0)
        for object in range(num_objects):
            left = torch.randint(0, im_w - 40, (1, )).item()
            upper = torch.randint(0, im_h - 40, (1, )).item()
            w = torch.randint(20, im_w - left, (1, )).item()
            h = torch.randint(20, im_h - upper, (1, )).item()
            area = w * h
            annotations.append({
                'id': object,
                'image_id': idx,
                'category_id': torch.randint(20, (1, )).item(),
                'bbox': [left, upper, w, h],
                'area': area,
                'segmentation': [],
                'iscrowd': 0
            })
        target = {"image_id": idx, "annotations": annotations}
        targets.append(target)
        idx += 1
    bsize = len(batch)

    try:
        encoding = feature_extractor(images=batch,
                                     annotations=targets,
                                     return_tensors='pt')
    except Exception as err:
        print('Encoding of batch failed.')
        raise

    try:
        with torch.no_grad():
            output = model(encoding["pixel_values"], encoding["pixel_mask"])
    except Exception as err:
        print('Forward pass failed.')
        raise

    assert len(output.logits) == bsize, 'Output does not match batch size'

    for logits in output.logits:
        assert logits.shape[1] == 21, 'Wrong number of classes in output'

    try:
        outout = model.training_step(encoding, batch_idx)
    except Exception as err:
        print('Training step failed.')
        raise

    try:
        with torch.no_grad():
            output = model.validation_step(encoding, batch_idx)
    except Exception as err:
        print('Validation step failed.')
        raise
