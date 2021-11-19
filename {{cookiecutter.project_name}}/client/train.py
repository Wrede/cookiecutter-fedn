from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import yaml
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from data.read_data import read_data
import os


def train(model,data,settings):
    """
    Training function which will be called upon model update requests
    from the combiner

    :param model: The latest global model, see '__main__'
    :type model: User defined
    :param data: Traning data
    :type data: User defined
    :param settings: Hyper-parameters settings
    :type settings: dict
    :return: Trained/updated model
    :rtype: User defined
    """
    print("-- RUNNING TRAINING --", flush=True)

    #CODE TO READ DATA
    
    #EXAMPLE
    #(x_train, y_train) = read_data(data, trainset=True)

    #CODE FOR START TRAINING
    #EXAMPLE (Tensoflow)
    #model.fit(x_train, y_train, batch_size=settings['batch_size'], epochs=settings['epochs'], verbose=1)

    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    #READ HYPER_PARAMETER SETTINGS FROM YAML FILE
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    #CREATE THE SEED MODEL AND UPDATE WITH LATEST WEIGHTS
    from fedn.utils.kerashelper import KerasHelper
    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    from models.model import create_seed_model
    model = create_seed_model()
    #EXAMPLE (HOW TO SET WEIGHTS OF A MODEL DIFFERS BETWEEN LIBRARIES)
    model.set_weights(weights)

    #CALL TRAINING FUNCTION AND GET UPDATED MODEL
    model = train(model,'../data/mnist.npz',settings)

    #SAVE/SEND MODEL
    helper.save_model(model.get_weights(),sys.argv[2])
