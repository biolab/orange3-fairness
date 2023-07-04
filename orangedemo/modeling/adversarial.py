from typing import Optional
from Orange.base import Learner, Model
from Orange.data import Table, Storage, Domain, table_from_frame

from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

from orangedemo.utils import table_to_standard_dataset, contains_fairness_attributes, MISSING_FAIRNESS_ATTRIBUTES

import numpy as np

# This gets called after the model is created and fitted
# It is stored so we can use it to predict on new data
class AdversarialDebiasingModel(Model):
    def __init__(self, model, domain):
        super().__init__()
        self._model = model
        self._domain = domain
        self.params = vars()

    def predict(self, data):
        if isinstance(data, Table):
            # For creating the standard dataset we need to know the encoding the table uses for the class variable
            # The encoding is ordinal and is the same as the orcer of values in the domain
            if not data.domain.class_var:
                data.domain.class_var = self._domain.class_var
            standard_dataset,_,_ = table_to_standard_dataset(data)
            predictions = self._model.predict(standard_dataset)
            # Create a array of scores with a column for each class the first column is the predictions.scores and the second column is 1 - predictions.scores
            # TODO: Check if the order of the columns is always correct
            second_column = 1 - predictions.scores
            second_column = second_column.reshape(-1, 1)  # reshape to (N, 1)
            scores = np.hstack((predictions.scores, second_column))

            return np.squeeze(predictions.labels, axis=1), scores
        else:
            print("Data is not a table")
    
    def predict_storage(self, data):
            if isinstance(data, Table):
                return self.predict(data)
            else:
                print("Data is not a table")
    
    def __call__(self, data, ret=Model.Value):
        return self.predict_storage(data)




class AdversarialDebiasingLearner(Learner):
    __returns__ = AdversarialDebiasingModel

    def __init__(self, preprocessors=None, **kwargs):
        self.params = vars()
        super().__init__(preprocessors=preprocessors)

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
        
    def fit(self, data: Table) -> AdversarialDebiasingModel:
        if not contains_fairness_attributes(data.domain):
            raise ValueError(MISSING_FAIRNESS_ATTRIBUTES)
        standardDataset, privileged_groups, unprivileged_groups = table_to_standard_dataset(data)


        # Create a new session and reset the default graph
        # Eager execution mea
        tf.disable_eager_execution()
        tf.reset_default_graph()
        if tf.get_default_session() is not None:
            tf.get_default_session().close()
        sess = tf.Session()

        print(f"params: {self.params['kwargs']}")

        # Create a model using the parameters from the widget and fit it to the data
        # **self.params["kwargs"] unpacks the dictionary self.params["kwargs"] into keyword arguments
        model = AdversarialDebiasing(**self.params["kwargs"], unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, sess=sess, scope_name="adversarial_debiasing")
        model = model.fit(standardDataset)
        return AdversarialDebiasingModel(model=model, domain=data.domain)
    
    def __call__(self, data, progress_callback=None):
        m = super().__call__(data, progress_callback)
        m.params = self.params
        return m
    
