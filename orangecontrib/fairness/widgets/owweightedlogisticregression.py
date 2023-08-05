from Orange.base import Learner
from Orange.classification.logistic_regression import LogisticRegressionLearner

import Orange.widgets.model.owlogisticregression


class WeightedLogisticRegressionLearner(LogisticRegressionLearner):
    """
    A class used to create a LogisticRegressionLearner which can
    use instance weights in the training and prediction process
    """

    def _fit_model(self, data):
        """
        A override of the _fit_model function of the LogisticRegressionLearner
        class which allows the use of instance weights
        """
        if type(self).fit is Learner.fit:
            return self.fit_storage(data)
        else:
            X, Y, W = data.X, data.Y, data.W if data.has_weights() else None
            if "weights" in map(lambda a: a.name, data.domain.metas):
                return self.fit(X, Y, W=data[:, "weights"].metas[:, 0])
            return self.fit(X, Y, W)


class OWWeightedLogisticRegression(Orange.widgets.model.owlogisticregression.OWLogisticRegression):
    """A class used to create a widget which uses the WeightedLogisticRegressionLearner"""

    name = "Weighted Logistic Regression"
    description = (
        "The logistic regression classification algorithm modification with "
        "LASSO (L1) or ridge (L2) regularization, that can use instance weights "
        "in the training and prediction process."
    )
    icon = "icons/weighted_log_reg.svg"
    priority = 50
    keywords = "weighted logistic regression"
    replaces = []

    LEARNER = WeightedLogisticRegressionLearner
