# -*- coding: utf-8 -*-
import logging
import os
import shutil
from pathlib import Path

import click
import requests
from dotenv import find_dotenv, load_dotenv

from src.data.dataset_utils import generate_new_annotation_file


@click.command()
def main():
    """
    Root should be mlops-project
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    """

    logger = logging.getLogger(__name__)

    if not os.path.exists("data"):
        logger.info("no /data directory found, creating...")
        os.mkdir("data")

    if not os.path.exists(os.path.join(os.getcwd(), "data/train")):
        logger.info("downloading PASCAL VOC 2012 (might take a few minutes)...")
        url = "https://public.roboflow.com/ds/TuRdHOyzfR?key=mVfrbTRBIf"
        r = requests.get(url, allow_redirects=True)
        os.chdir(os.path.join(os.getcwd(), "data"))
        open("pascal.zip", "wb").write(r.content)
        shutil.unpack_archive("pascal.zip", os.getcwd())
        os.remove("pascal.zip")
        os.chdir(os.pardir)
    else:
        logger.info("found PASCAL VOC 2012")

    logger.info("generating annotation files")
    generate_new_annotation_file("train")
    generate_new_annotation_file("valid")
    logger.info("done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
