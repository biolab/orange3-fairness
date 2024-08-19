"""
This module contains the PostprocessingModel and PostprocessingLearner classes 

These are used to create and fit the model and postprocessor and create the PostprocessingModel.
"""

import numpy as np

from Orange.base import Learner, Model
from Orange.data import Table
from Orange.evaluation import CrossValidation

from aif360.algorithms.postprocessing import EqOddsPostprocessing

from orangecontrib.fairness.widgets.utils import (
    table_to_standard_dataset,
    contains_fairness_attributes,
    MISSING_FAIRNESS_ATTRIBUTES,
)


class PostprocessingModel(Model):
    """
    Model created and fitted by the PostprocessingLearner

    Used to predict on new data and to postprocess the predictions

    Attributes:
    - model (Model): The model used to make predictions
    - postprocessor (EqOddsPostprocessing): The postprocessor used to postprocess the predictions
    """

    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.params = vars()

    def predict(self, data):
        """
        Method used to preprocess, predict and postprocess on new data.

        First we get the predictions from the model, then we use the
        postprocessor on the predictions, and finally we create dummy scores
        which are used to aproximate the scores of the postprocessed predictions.
        """
        if isinstance(data, Table):
            predictions, _ = self.model(data, ret=Model.ValueProbs)

            standard_dataset, _, _ = table_to_standard_dataset(data)
            standard_dataset_pred = standard_dataset.copy(deepcopy=True)
            standard_dataset_pred.labels = predictions.reshape(-1, 1)

            standard_dataset_pred_transf = self.postprocessor.predict(
                standard_dataset_pred
            )

            # Create dummy scores from predictions
            # (if the predictions are 0 or 1, the scores will be 0 or 1)
            scores = np.zeros((len(standard_dataset_pred_transf.labels), 2))
            scores[:, 1] = standard_dataset_pred_transf.labels.ravel()
            scores[:, 0] = 1 - standard_dataset_pred_transf.labels.ravel()

            return np.squeeze(standard_dataset_pred_transf.labels, axis=1), scores

    def predict_storage(self, data):
        if isinstance(data, Table):
            return self.predict(data)
        else:
            raise TypeError("Data is not of type Table")

    def __call__(self, data, ret=Model.Value):
        return super().__call__(data, ret)


class PostprocessingLearner(Learner):
    """
    Subclass used to create and fit the model and postprocessor and create the PostprocessingModel

    Attributes:
    - learner (Learner): The learner used to create the model
    - preprocessors (list): The preprocessors used to preprocess the data
    - repeatable (bool): If the model should be repeatable
    - callback (function): The callback used to interrupt the widget
    - seed (int): The seed used to make the model repeatable
    - params (dict): The parameters used in the __call__ method
    """

    __returns__ = PostprocessingModel

    def __init__(self, learner, preprocessors=None, repeatable=None):
        super().__init__(preprocessors=preprocessors)
        self.learner = learner
        self.callback = None
        self.seed = 42 if repeatable else None
        self.params = vars()

    def incompatibility_reason(self, domain):
        """
        Method used to check if the domain contains the fairness attributes
        """
        if not contains_fairness_attributes(domain):
            return MISSING_FAIRNESS_ATTRIBUTES

    # Fit storage and fit functions were modified to use a Table/Storage object
    # This is because it's the easiest way to get the domain, and meta attributes
    def fit_storage(self, data):
        if isinstance(data, Table):
            self.fit(data)
        else:
            raise TypeError("Data is not of type Table")

    def _fit_model(self, data):
        if type(self).fit is Learner.fit:
            return self.fit_storage(data)
        else:
            return self.fit(data)

    def fit(self, data):
        """
        Method used to preprocess the data, fit the model and the postprocessor
        """
        if isinstance(data, Table):
            if not contains_fairness_attributes(data.domain):
                raise ValueError(MISSING_FAIRNESS_ATTRIBUTES)

            # Fit the model to the data
            # the callback is currently not used for progress but to allow
            # the user to interrupt the widget while the model is training
            model = self.learner(data, self.callback)

            # Use cross validation to get the predictions, we do this to avoid having to use
            # a train/validation split to get the predictions required to fit the postprocessor
            cv = CrossValidation(k=5)
            # Including the callback in the cross validation will allow the user to
            # interrupt the widget when it's in the cross validation phase
            res = cv(data, [self.learner], callback=self.callback)
            predictions = res.predicted[0]
            row_indices = res.row_indices
            predictions = predictions[np.argsort(row_indices)]

            # Get the predictions which will be used to fit the postprocessor
            (
                standard_dataset,
                privileged_groups,
                unprivileged_groups,
            ) = table_to_standard_dataset(data)
            standard_dataset_pred = standard_dataset.copy(deepcopy=True)
            standard_dataset_pred.labels = predictions

            # Create and fit the postprocessor to the predictions
            postprocessor = EqOddsPostprocessing(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
                seed=self.seed,
            )
            postprocessor.fit(standard_dataset, standard_dataset_pred)
            return PostprocessingModel(model, postprocessor)
        else:
            raise TypeError("Data is not of type Table")

    def __call__(self, data, progress_callback=None):
        self.callback = progress_callback
        # By adding this line, the progress will "work" for learners with the
        # .callback attribute but it may cause problems for learners without it
        # self.learner.callback = progress_callback
        model = super().__call__(data)
        model.params = self.params
        return model
