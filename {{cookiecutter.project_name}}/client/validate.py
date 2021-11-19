import sys
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from data.read_data import read_data
import json
from sklearn import metrics
import os
import yaml
import numpy as np

def validate(model,data):
    """
    Validation function which will be called upon model validation requests
    from the combiner.

    :param model: The latest global model, see '__main__'
    :type model: User defined
    :param data: The data used for validation, could include both training and test/validation data
    :type data: User defined
    :return: Model scores from the validation
    :rtype: dict
    """
    print("-- RUNNING VALIDATION --", flush=True)

    #CODE TO READ DATA
    
    #EXAMPLE
    #(x_train, y_train) = read_data(data, trainset=True)

    #EXAMPLE
    #(x_test, y_test) = read_data(data, trainset=False)
     
    try:
        #CODE HERE FOR OBTAINING VALIDATION SCORES 
        
        #EXAMPLE
        #model_score = model.evaluate(x_train, y_train, verbose=0)
        #model_score_test = model.evaluate(x_test, y_test, verbose=0)
        #y_pred = model.predict(x_test)
        #y_pred = np.argmax(y_pred, axis=1)
        #clf_report = metrics.classification_report(y_test.argmax(axis=-1),y_pred)

    except Exception as e:
        print("failed to validate the model {}".format(e),flush=True)
        raise

    #PUT SCORES AS VALUES FOR CORRESPONDING KEYS (CHANGE VARIABLES):
    report = {
                "classification_report": clf_report,
                "training_loss": model_score[0],
                "training_accuracy": model_score[1],
                "test_loss": model_score_test[0],
                "test_accuracy": model_score_test[1],
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

if __name__ == '__main__':

    #READS THE LATEST WEIGHTS FROM GLOBAL MODEL (COMBINER)
    
    from fedn.utils.kerashelper import KerasHelper
    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    #CREATE THE SEED MODEL AND UPDATE WITH LATEST WEIGHTS
    from models.model import create_seed_model
    model = create_seed_model()
    #EXAMPLE (HOW TO SET WEIGHTS OF A MODEL DIFFERS BETWEEN LIBRARIES)
    model.set_weights(weights)
    
    #START VALIDATION
    report = validate(model,'../data/mnist.npz')

    #SAVE/SEND SCORE REPORT
    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(report))
