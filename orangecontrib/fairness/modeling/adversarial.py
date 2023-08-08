import numpy as np

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


# This gets called after the model is created and fitted
# It is stored so we can use it to predict on new data
class AdversarialDebiasingModel(Model):
    """Model created and fitted by the AdversarialDebiasingLearner, which is used to predict on new data"""

    def __init__(self, model):
        super().__init__()
        self._model = model
        self.params = vars()

    def predict(self, data):
        """Function used to predict on new data"""
        if isinstance(data, Table):
            standard_dataset, _, _ = table_to_standard_dataset(data)
            predictions = self._model.predict(standard_dataset)

            # Array of scores with a column of scores for each class
            scores = np.hstack(
                (predictions.scores, (1 - predictions.scores).reshape(-1, 1))
            )

            temp = np.squeeze(predictions.labels, axis=1), scores
            return temp
        else:
            raise TypeError("Data is not of type Table")

    def predict_storage(self, data):
        if isinstance(data, Table):
            return self.predict(data)
        else:
            raise TypeError("Data is not of type Table")

    def __call__(self, data, ret=Model.Value):
        return super().__call__(data, ret)


class AdversarialDebiasingLearner(Learner):
    """Learner subclass used to create and fit the AdversarialDebiasingModel"""

    __returns__ = AdversarialDebiasingModel
    # List of preprocessors, these get applied when the __call__ function is called
    preprocessors = [Normalize()]
    callback = None

    def __init__(self, preprocessors=None, **kwargs):
        self.params = vars()
        super().__init__(preprocessors=preprocessors)

    def _calculate_total_runs(self, data):
        """Function used to calculate the total number of runs the learner will perform on the data"""
        # This is need to calculate and display the progress of the training
        num_epochs = self.params["kwargs"]["num_epochs"]
        batch_size = self.params["kwargs"]["batch_size"]
        num_instances = len(data)
        num_batches = np.ceil(num_instances / batch_size)
        total_runs = num_epochs * num_batches
        return total_runs

    def incompatibility_reason(self, domain):
        """Function used to check if the domain is compatible with the learner (contains fairness attributes)"""
        if not contains_fairness_attributes(domain):
            return MISSING_FAIRNESS_ATTRIBUTES

    def fit_storage(self, data):
        return self.fit(data)

    def _fit_model(self, data):
        if type(self).fit is Learner.fit:
            return self.fit_storage(data)
        else:
            return self.fit(data)

    # Fit storage and fit functions were modified to use a Table/Storage object
    # This is because it's the easiest way to get the domain, and meta attributes
    # TODO: Should I use the X,Y,W format instead of the table format ? (Same for the model)
    def fit(self, data: Table) -> AdversarialDebiasingModel:
        (
            standard_dataset,
            privileged_groups,
            unprivileged_groups,
        ) = table_to_standard_dataset(data)

        tf.disable_eager_execution()
        tf.reset_default_graph()
        if tf.get_default_session() is not None:
            tf.get_default_session().close()
        sess = CallbackSession(
            callback=self.callback, total_runs=self._calculate_total_runs(data)
        )

        # Create a model using the parameters from the widget and fit it to the data
        model = AdversarialDebiasing(
            **self.params["kwargs"],
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            sess=sess,
            scope_name="adversarial_debiasing"
        )
        sess.enable_callback()
        model = model.fit(standard_dataset)
        print(f"Weights training: {standard_dataset.instance_weights}")
        sess.disable_callback()
        return AdversarialDebiasingModel(model=model)

    def __call__(self, data, progress_callback=None):
        """Call method for AdversarialDebiasingLearner, in the superclass it calls the _fit_model function (and other things)"""
        self.callback = progress_callback
        model = super().__call__(data, progress_callback)
        model.params = self.params
        return model


class CallbackSession(tf.Session):
    """Subclass of tensorflow session with callback functionality for progress tracking and displaying"""

    def __init__(self, target="", graph=None, config=None, callback=None, total_runs=0):
        super().__init__(target=target, graph=graph, config=config)
        self.callback = callback
        self.run_count = 0
        self.callback_enabled = False
        self.total_runs = total_runs

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """A overridden run function which calls the callback function and calculates the progress"""
        # To calculate the progress using these ways we need to know the number of expected
        # calls to the callback function and count how many times it has been called
        self.run_count += 1
        progress = (self.run_count / self.total_runs) * 100
        if self.callback_enabled and self.callback:
            self.callback(progress)

        return super().run(
            fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata
        )

    def enable_callback(self):
        """Enable callback method for the model fitting fase"""
        self.callback_enabled = True

    def disable_callback(self):
        """Disable callback method for the model prediction fase"""
        self.callback_enabled = False