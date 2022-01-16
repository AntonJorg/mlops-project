
# mlops-project
Project work for 02476 Machine Learning Operations

## Goal
We plan to use the huggingface transformer framework to train a DETR model for object detection on the Pascal VOC 2012 Dataset. We wish to make our results reproducible and make the simplest and most readable implementation possible.

## Framework
The tranformer framework supports the DETR object detection model and pretrained weights can easily be applied. We plan to use the pretraining feature as well as the DETR model classes to obtain a model which can readily be finetuned on a new dataset. The main motivation for using the transformer framework is to take care of the hard parts of the implementation, such that we do not have to think about model architecture and efficiency directly.

## Model
DETR is an end-to-end trainable tranformer neural network for object detection. It solves some of the issues of previous methods, mainly by being simpler to implement.

## Data
The Pascal VOC 2012 Dataset is an object detection dataset with 17112 images and 20 classes and is commonly used as a benchmark for image detection models.

# User guide
In this section, a short guide to get started using the project is provided.

To get started, use the pip command
```
pip install -r requirements.txt
```
to install any and all dependencies needed for the project.

To download and process the Pascal VOC 2012 Dataset, use
```
make data
```

To train a new model use
```
make train
```

To make a prediction using a trained model use
```
python src/models/predict_model.py {path to model weights} {path OR url to images}
```

# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

