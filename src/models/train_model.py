import logging

import hydra
from dotenv import find_dotenv, load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from src.data.dataset_utils import get_dataloaders
from src.models.model import DetrPascal

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml")
def main(config):
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")

    hparams = config.experiment

    seed_everything(hparams.seed, workers=True)

    model = DetrPascal(
        lr=hparams.lr,
        lr_backbone=hparams.lr_backbone,
        weight_decay=hparams.weight_decay,
    )
    # Batch size of 2 * gpus recommended here:
    # https://huggingface.co/docs/transformers/model_doc/detr
    train_dataloader, val_dataloader = get_dataloaders(
        to_absolute_path("data"), hparams.batch_size
    )

    # Train
    trainer = Trainer(
        gpus=hparams.gpus, amp_backend=hparams.amp_backend, amp_level=hparams.amp_level
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
