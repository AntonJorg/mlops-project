import logging
import os

import hydra
import wandb
from dotenv import find_dotenv, load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from os.path import join, dirname
from src.data.dataset_utils import get_dataloaders
from src.models.model import DetrPascal

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config.yaml")
def main(config):
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    loggers = [TensorBoardLogger("lightning_logs/", name="")]
    if config.wandb:
        load_dotenv('.env')
        api_key = os.getenv("WANDB_API_KEY")
        project = os.getenv("WANDB_PROJECT")
        entity = os.getenv("WANDB_ENTITY")
        if not (api_key or project):
            raise EnvironmentError(
                "Trying to use wandb logging without WANDB_API_KEY or WANDB_PROJECT defined in .env")
        wandb.login(key=api_key)
        wandb.init(project=project,entity=entity)
        loggers.append(WandbLogger())

    hparams = config.experiment

    checkpoint_callback = ModelCheckpoint(**hparams.checkpoint_callback)


    seed_everything(hparams.seed, workers=True)

    model = DetrPascal(**hparams.model)
    # Batch size of 2 * gpus recommended here:
    # https://huggingface.co/docs/transformers/model_doc/detr
    train_dataloader, val_dataloader = get_dataloaders(
        to_absolute_path("data"), hparams.batch_size
    )

    # Train
    trainer = Trainer(default_root_dir=config.default_root_dir,
        logger=loggers,
        callbacks=[checkpoint_callback],
        **hparams.trainer
        )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
