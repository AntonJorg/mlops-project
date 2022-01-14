import click


@click.command()
@click.argument('load_model_from', type=click.Path(exists=True))
# Path to preloaded images for prediction
@click.argument('images_from', type=click.Path(exists=True))  
def predict(load_model_from, images_from):
    """Predicts objects in the provided images.
    
    Arguments:
        load_model_from {str}: path to model weights
        images {str}: path to images for prediction
    """
    return None

if __name__ == '__main__':
    predict()
