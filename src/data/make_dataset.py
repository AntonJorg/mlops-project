# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torchvision
import os
import requests
import shutil
import os
import json
from src.data.dataset_utils import generate_new_annotation_file

@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Root should be mlops-project
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    """

    if not os.path.exists(os.path.join(os.getcwd(),'data/train')):
        url = 'https://public.roboflow.com/ds/TuRdHOyzfR?key=mVfrbTRBIf'
        r = requests.get(url, allow_redirects=True)
        os.chdir(os.path.join(os.getcwd(),'data'))
        open('pascal.zip', 'wb').write(r.content)
        shutil.unpack_archive('pascal.zip', os.getcwd())
        os.remove('pascal.zip')
        os.chdir(os.pardir)

    generate_new_annotation_file('train')
    generate_new_annotation_file('valid')


    logger = logging.getLogger(__name__)

    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

