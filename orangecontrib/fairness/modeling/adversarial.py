"""
This module contains the AdversarialDebiasingLearner and AdversarialDebiasingModel classes 
which are used to create and fit the AdversarialDebiasing model from the aif360 library.
"""

import numpy as np

from Orange.base import Learner, Model
from Orange.data import Table
from Orange.preprocess import Normalize

from orangecontrib.fairness.widgets.utils import (
    table_to_standard_dataset,
    contains_fairness_attributes,
    MISSING_FAIRNESS_ATTRIBUTES,
    is_tensorflow_installed,
)

if is_tensorflow_installed():
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    import tensorflow.compat.v1 as tf
else:
    AdversarialDebiasing = None
    tf = None


# This gets called after the model is created and fitted
# It is stored so we can use it to predict on new data
class AdversarialDebiasingModel(Model):
    """
    Model created and fitted by the AdversarialDebiasingLearner, used to predict on new data.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def predict(self, data):
        """
        Method used to 'preprocess', predict on new data and 'postprocess' the predictions.

        Args:
            data (Table): The data to predict on.
        """
        if isinstance(data, Table):
            standard_dataset, _, _ = table_to_standard_dataset(data)
            predictions = self._model.predict(standard_dataset)

            # Array of scores with a column of scores for each class
            # The scores given by the model are always for the favorable class
            # If the favorable class is 1 then the scores need to be flipped or
            # else the AUC will be "reversed"
            # (the first column is 1 - scores and the second column is scores)
            if standard_dataset.favorable_label == 0:
                scores = np.hstack(
                    (predictions.scores, (1 - predictions.scores).reshape(-1, 1))
                )
            else:
                scores = np.hstack(
                    ((1 - predictions.scores).reshape(-1, 1), predictions.scores)
                )

            predictions_scores = np.squeeze(predictions.labels, axis=1), scores
            return predictions_scores
        else:
            raise TypeError("Data is not of type Table")

    def predict_storage(self, data):
        if isinstance(data, Table):
            return self.predict(data)
        else:
            raise TypeError("Data is not of type Table")

    def __call__(self, data, ret=Model.Value):
        return super().__call__(data, ret)


if is_tensorflow_installed():

    class AdversarialDebiasingLearner(Learner):
        """
        Learner subclass used to create and fit the AdversarialDebiasingModel

        Attributes:
            preprocessors (list): List of preprocessors, applied when __call__ function is called
            callback (function): Callback function used to track the progress of the model fitting

        Args:
            preprocessors (list): List of preprocessors to apply to the data before fitting a model
            classifier_num_hidden_units (int): Number of hidden units in the classifier
            num_epochs (int): Number of epochs to train the model
            batch_size (int): Batch size used to train the model
            debias (bool): Whether to debias the model
            adversary_loss_weight (float): Weight of the adversary loss
            seed (int): Seed used to initialize the model
        """

        __returns__ = AdversarialDebiasingModel
        preprocessors = [Normalize()]
        callback = None

        def __init__(
            self,
            preprocessors=None,
            classifier_num_hidden_units=100,
            num_epochs=50,
            batch_size=128,
            debias=True,
            adversary_loss_weight=0.1,
            seed=-1,
        ):
            super().__init__(preprocessors=preprocessors)
            self.params = vars()

            self.model_params = {
                "classifier_num_hidden_units": classifier_num_hidden_units,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "debias": debias,
                "adversary_loss_weight": adversary_loss_weight,
                **({"seed": seed} if seed != -1 else {}),
            }

        def _calculate_total_runs(self, data):
            """
            Method for calculating the total number of runs the learner will perform on the data

            Used to calculate and display the progress of the training.
            """
            num_epochs = self.params["num_epochs"]
            batch_size = self.params["batch_size"]
            num_instances = len(data)
            num_batches = np.ceil(num_instances / batch_size)
            total_runs = num_epochs * num_batches
            return total_runs

        def incompatibility_reason(self, domain):
            """
            Method used to check if the domain is compatible with the learner.

            The domain is compatible if it contains the fairness attributes.
            """
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
                **self.model_params,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
                sess=sess,
                scope_name="adversarial_debiasing"
            )
            sess.enable_callback()
            model = model.fit(standard_dataset)
            sess.disable_callback()
            return AdversarialDebiasingModel(model=model)

        def __call__(self, data, progress_callback=None):
            """
            Call method for AdversarialDebiasingLearner

            In the superclass it calls the _fit_model function (and other things)
            """
            self.callback = progress_callback
            model = super().__call__(data, progress_callback)
            model.params = self.params
            return model

    class CallbackSession(tf.Session):
        """
        Subclass of tensorflow session.

        It adds callback functionality for progress tracking and displaying.

        Attributes:
            callback (function): Callback function used to track the progress of the model fitting
            run_count (int): Number of times the run function has been called
            callback_enabled (bool): Flag to enable or disable the callback function
            total_runs (int): Total number of runs the session will perform
        """

        def __init__(
            self, target="", graph=None, config=None, callback=None, total_runs=0
        ):
            super().__init__(target=target, graph=graph, config=config)
            self.callback = callback
            self.run_count = 0
            self.callback_enabled = False
            self.total_runs = total_runs

        def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
            """
            A overridden run function which calls the callback function and calculates the progress

            To calculate the progress using these ways we need to know the number of expected
            calls to the callback function and count how many times it has been called.
            """

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

else:

    class AdversarialDebiasingLearner(Learner):
        """Dummy class used if tensorflow is not installed"""

        __returns__ = Model
