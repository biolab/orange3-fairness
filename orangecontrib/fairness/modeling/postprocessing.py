import numpy as np

from Orange.base import Learner, Model
from Orange.data import Table

from aif360.algorithms.postprocessing import EqOddsPostprocessing


from orangecontrib.fairness.widgets.utils import (
    table_to_standard_dataset,
    contains_fairness_attributes,
    MISSING_FAIRNESS_ATTRIBUTES,
)


class PostprocessingModel(Model):
    def __init__(self, model, postprocessor, learner):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.learner = learner
        self.params = vars()

    def predict(self, data):
        if isinstance(data, Table):
            # Normalize the data
            data = self.learner.preprocess(data)
            # Get the predictions and scores from the model
            predictions, scores = self.model.predict_storage(data)

            # For creating the standard dataset we need to know the encoding the table uses for the class variable, the encoding is ordinal and is the same as the order of values in the domain
            if not data.domain.class_var:
                data.domain.class_var = self.original_domain.class_var
            standard_dataset, _, _ = table_to_standard_dataset(data)
            standard_dataset_pred = standard_dataset.copy(deepcopy=True)
            standard_dataset_pred.labels = predictions.reshape(-1, 1)

            # Postprocess the predictions
            standard_dataset_pred_transf = self.postprocessor.predict(
                standard_dataset_pred
            )

            # Return the postprocessed predictions and the scores
            return np.squeeze(standard_dataset_pred_transf.labels, axis=1), scores

    def predict_storage(self, data):
        if isinstance(data, Table):
            return self.predict(data)
        else:
            raise TypeError("Data is not of type Table")

    def __call__(self, data, ret=Model.Value):
        return self.predict_storage(data)


class PostprocessingLearner(Learner):
    __returns__ = PostprocessingModel

    def __init__(self, learner, preprocessors=None, repeatable=None):
        super().__init__(preprocessors=preprocessors)
        self.learner = learner
        self.seed = 42 if repeatable else None
        self.params = vars()

    def incompatibility_reason(self, domain):
        if not contains_fairness_attributes(domain):
            return MISSING_FAIRNESS_ATTRIBUTES

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
        if isinstance(data, Table):
            if not contains_fairness_attributes(data.domain):
                raise ValueError(MISSING_FAIRNESS_ATTRIBUTES)
            # Normalize the data
            data = self.preprocess(data)

            # Fit the model TODO: Split the data into train and test data so we can fit the postprocessor on the test data to avoid data leakage
            model = self.learner.fit_storage(data)
            # Because I use fit_storage instead of __call__ I need to set the original domain manually (this is needed for the postprocessor to be compatible with adversarial debiasing)
            model.original_domain = data.domain
            predictions, _ = model.predict_storage(
                data
            )  # Returns the predictions and the scores

            # Get the predictions which will be used to fit the postprocessor
            (
                standard_dataset,
                privileged_groups,
                unprivileged_groups,
            ) = table_to_standard_dataset(data)
            standard_dataset_pred = standard_dataset.copy(deepcopy=True)
            standard_dataset_pred.labels = predictions
            # Fit the postprocessor
            postprocessor = EqOddsPostprocessing(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
                seed=self.seed,
            )
            postprocessor.fit(standard_dataset, standard_dataset_pred)
            return PostprocessingModel(model, postprocessor, self)
        else:
            raise TypeError("Data is not of type Table")

    def __call__(self, data):
        model = super().__call__(data)
        model.params = self.params
        return model
