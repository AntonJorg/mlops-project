import requests
from PIL import Image
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import DetrForObjectDetection


MODEL_NAME = "mishig/tiny-detr-mobilenetsv3"


class DetrPascal(LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = DetrForObjectDetection.from_pretrained(
            MODEL_NAME, num_labels=20, ignore_mismatched_sizes=True
        )
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss.item(), batch_size=len(batch["pixel_values"]))
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), batch_size=len(batch["pixel_values"]))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss.item(), batch_size=len(batch["pixel_values"]))
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), batch_size=len(batch["pixel_values"]))

        return loss

    def configure_optimizers(self):
        """
        Manual Docstring Test
        """
        param_dicts = [
            {
                "params": [
                    p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
