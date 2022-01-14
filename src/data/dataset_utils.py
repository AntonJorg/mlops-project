import json
import os
from typing import Optional, Tuple

import torchvision
from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor


class PascalDataset(torchvision.datasets.CocoDetection):
    """Dataset class for the Pascal VOC data.

    Attributes:
        feature_extractor: Instance of DetrFeatureExtractor. Used for preprocessing of images.
    """

    def __init__(
        self, dataset_path: str, feature_extractor: DetrFeatureExtractor, train=True
    ):
        img_folder = os.path.join(dataset_path, "train" if train else "valid")
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super().__getitem__(idx)

        # preprocess image and target
        # (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target


class Collator:
    """
    This class is passed to the 'collate_fn' argument of data loaders of a PascalDataset,
    to perform batch-wise padding of the input features.
    """

    def __init__(self):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50"
        )

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.feature_extractor.pad_and_create_pixel_mask(
            pixel_values, return_tensors="pt"
        )
        labels = [item[1] for item in batch]
        batch = {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }
        return batch


def get_dataloaders(
    dataset_path: str, batch_size: int, cpu_count: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns data loaders for the training and validation set.

            Parameters:
                    batch_size: Batch size for the data loaders.
                    cpu_count: Determines 'num_workers' for the data loaders. If None, defaults to os.cpu_count().
            Returns:
                    train_dataloader: Data loader for the training set.
                    val_dataloader: Data loader for the validation set.
    """
    collator = Collator()

    train_dataset = PascalDataset(
        dataset_path=dataset_path, feature_extractor=collator.feature_extractor
    )
    val_dataset = PascalDataset(
        dataset_path=dataset_path,
        feature_extractor=collator.feature_extractor,
        train=False,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=collator,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        collate_fn=collator,
    )

    return train_dataloader, val_dataloader


def new_annotation_structure(anno_dict, dataset):
    annotations = anno_dict["annotations"]
    images = anno_dict["images"]
    annotations_new = []
    i = 0
    while i < len(annotations):
        image_id_prev = annotations[i]["image_id"]
        image_path = os.path.join("data", dataset, images[image_id_prev]["file_name"])
        dict_t = {
            "image_id": image_id_prev,
            "image_location": image_path,
            "annotations": [],
        }
        while image_id_prev == annotations[i]["image_id"]:
            dict_t["annotations"].append(annotations[i])
            i += 1
            if i >= len(annotations):
                break
        annotations_new.append(dict_t)
    return annotations_new


def generate_new_annotation_file(dataset=None):
    path = os.path.join(os.getcwd(), "data", dataset)
    file = open(os.path.join(path, "_annotations.coco.json"), "r")
    d = json.load(file)
    file.close()
    annotations = new_annotation_structure(d, dataset)
    annotation_path = os.path.join(path, "new_annotations.json")
    if os.path.exists(annotation_path):
        os.remove(annotation_path)
    annotation_file = open(annotation_path, "w")
    json.dump(annotations, annotation_file)


if __name__ == "__main__":
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = PascalDataset("data", feature_extractor=feature_extractor)
    val_dataset = PascalDataset(
        "data", feature_extractor=feature_extractor, train=False
    )

    print("Training images:", len(train_dataset))
    print("Test images    :", len(val_dataset))
    cats = train_dataset.coco.cats
    print("Classes (incuding No Object):", len(cats))
