from Orange.base import Learner, Model
from Orange.data import Table
from Orange.preprocess import Normalize

from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

from orangecontrib.fairness.widgets.utils import (
    table_to_standard_dataset,
    contains_fairness_attributes,
    MISSING_FAIRNESS_ATTRIBUTES,
)

import numpy as np


# This gets called after the model is created and fitted
# It is stored so we can use it to predict on new data
class AdversarialDebiasingModel(Model):
    def __init__(self, model, learner):
        super().__init__()
        self._model = model
        self.learner = learner
        self.params = vars()

    def predict(self, data):
        if isinstance(data, Table):
            # Normalize the data
            data = self.learner.preprocess(data)
            # For creating the standard dataset we need to know the encoding the table uses for the class variable, the encoding is ordinal and is the same as the order of values in the domain
            if not data.domain.class_var:
                data.domain.class_var = self.original_domain.class_var
            standard_dataset, _, _ = table_to_standard_dataset(data)
            predictions = self._model.predict(standard_dataset)

            # Create a array of scores with a column for each class the first column is the predictions.scores and the second column is 1 - predictions.scores
            scores = np.hstack(
                (predictions.scores, (1 - predictions.scores).reshape(-1, 1))
            )

            return np.squeeze(predictions.labels, axis=1), scores
        else:
            raise TypeError("Data is not of type Table")

    def predict_storage(self, data):
        if isinstance(data, Table):
            return self.predict(data)
        else:
            raise TypeError("Data is not of type Table")

    def __call__(self, data, ret=Model.Value):
        return self.predict_storage(data)


class AdversarialDebiasingLearner(Learner):
    __returns__ = AdversarialDebiasingModel
    # List of preprocessors, these get applied when the __call__ function is called
    preprocessors = [
        Normalize()
    ]
    callback = None

    def __init__(self, preprocessors=None, **kwargs):
        self.params = vars()
        super().__init__(preprocessors=preprocessors)

    def _calculate_total_runs(self, data):
        """Function used to calculate the total number of runs the learner will perform on the data"""
        
        # Retrieve the number of epochs and batch size directly from params
        num_epochs = self.params["kwargs"]['num_epochs']
        batch_size = self.params["kwargs"]['batch_size']
        num_instances = len(data)

        # Compute the total number of batches
        num_batches = np.ceil(num_instances / batch_size)

        # Total number of runs is the product of the number of epochs and the number of batches
        total_runs = num_epochs * num_batches

        return total_runs


    def incompatibility_reason(self, domain):
        if not contains_fairness_attributes(domain):
            return MISSING_FAIRNESS_ATTRIBUTES

    def fit_storage(self, data):
        return self.fit(data)

    def _fit_model(self, data):
        if type(self).fit is Learner.fit:
            return self.fit_storage(data)
        else:
            return self.fit(data)

    # Function responsible for fitting the learner to the data and creating a model
    # TODO: Should I use the X,Y,W format instead of the table format ?
    def fit(self, data: Table) -> AdversarialDebiasingModel:
        standardDataset, privileged_groups, unprivileged_groups = table_to_standard_dataset(data)
        # Create a new session and reset the default graph
        # Eager execution mea
        tf.disable_eager_execution()
        tf.reset_default_graph()
        if tf.get_default_session() is not None:
            tf.get_default_session().close()
        sess = CallbackSession(callback=self.callback, total_runs=self._calculate_total_runs(data))

        # Create a model using the parameters from the widget and fit it to the data
        # **self.params["kwargs"] unpacks the dictionary self.params["kwargs"] into keyword arguments
        model = AdversarialDebiasing(
            **self.params["kwargs"],
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            sess=sess,
            scope_name="adversarial_debiasing"
        )
        sess.enable_callback()
        model = model.fit(standardDataset)
        sess.disable_callback()
        return AdversarialDebiasingModel(model=model, learner=self)

    # This is called when using the learner as a function, in the superclass it uses the _fit_model function
    # Which creates a new model by calling the fit function
    def __call__(self, data, progress_callback=None):
        self.callback = progress_callback
        model = super().__call__(data, progress_callback)
        model.params = self.params
        return model
    



# Here I make a subclass of the tensorflow session (sess) and override the run function so it calls the callback function
# This allows me to calculate and display the progress of the training
# To calculate the progress using these ways we need to know the number of expected calls to the callback function and count how many times it has been called
class CallbackSession(tf.Session):
    def __init__(self, target='', graph=None, config=None, callback=None, total_runs=0):
        super().__init__(target=target, graph=graph, config=config)
        self.callback = callback
        self.run_count = 0
        self.total_runs = total_runs

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        self.run_count += 1
        progress = (self.run_count / self.total_runs) * 100
        if self.callback_enabled and self.callback:
            self.callback(progress)

        return super().run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
    
    # These two functions are needed so the callback can be enabled for the fit fase and disabled for the predict fase
    def enable_callback(self):
        self.callback_enabled = True

    def disable_callback(self):
        self.callback_enabled = False






