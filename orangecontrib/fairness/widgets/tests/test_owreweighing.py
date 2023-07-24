import os
import unittest
import numpy as np

from Orange.widgets.tests.base import WidgetTest
from Orange.classification.logistic_regression import LogisticRegressionLearner
from Orange.data import Table

from orangecontrib.fairness.widgets.owasfairness import OWAsFairness
from orangecontrib.fairness.widgets.owreweighing import OWReweighing
from orangecontrib.fairness.widgets.tests.utils import as_fairness_setup


class TestOWReweighing(WidgetTest):
    def setUp(self) -> None:
        self.test_data_path = "https://datasets.biolab.si/core/adult.tab"
        self.widget = self.create_widget(OWReweighing)
        self.as_fairness = self.create_widget(OWAsFairness)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.preprocessor))

    def test_incorrect_input_data(self):
        """Check that the widget displays an error message when the input data does not have the 'AsFairness' attributes"""
        test_data = Table(self.test_data_path)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.assertTrue(self.widget.Error.missing_fairness_data.is_shown())
        
    def test_output_data(self):
        """Check that the widget handles data correctly and adds the 'weights' column"""
        test_data = as_fairness_setup(self)
        self.send_signal(self.widget.Inputs.data, test_data)
        output_data = self.get_output(self.widget.Outputs.data)
        self.assertIsNotNone(output_data)
        self.assertIn("weights", output_data.domain)

    def test_preprocessor_output(self):
        """Check that the widget returns a working preprocessor"""
        test_data = as_fairness_setup(self)
        preprocessor = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsNotNone(preprocessor)
        preprocessed_data = preprocessor(test_data)
        self.assertIsNotNone(preprocessed_data)
        self.assertIn("weights", preprocessed_data.domain)

    def test_with_logistic_regression(self):
        """Check that the predictions of logistic regression on the original data and the preprocessed data are different"""

        ##############################################################################################################
        # Currently, this test fails because this implementation of LogisticRegressionLearner does not support weights
        ##############################################################################################################

        # test_data = as_fairness_setup(self)
        # self.send_signal(self.widget.Inputs.data, test_data)
        # preprocessed_data = self.get_output(self.widget.Outputs.data)
        # self.assertIsNotNone(preprocessed_data)

        # # print(f"Normal domain: {test_data.domain}")
        # # print(f"Preprocessed domain: {preprocessed_data.domain}")
        # # print(f"Weights: {preprocessed_data.metas[:5,:]}")

        # learner = LogisticRegressionLearner()
        # preprocessed_model = learner.fit_storage(preprocessed_data)
        # normal_model = learner.fit_storage(test_data)

        # (
        #     preprocessed_predictions,
        #     preprocessed_scores,
        # ) = preprocessed_model.predict_storage(test_data)
        # normal_predictions, normal_scores = normal_model.predict_storage(test_data)

        # self.assertFalse(
        #     np.array_equal(preprocessed_predictions, normal_predictions),
        #     "Preprocessed predictions should not equal normal predictions",
        # )
        # self.assertFalse(
        #     np.array_equal(preprocessed_scores, normal_scores),
        #     "Preprocessed scores should not equal normal scores",
        # )


if __name__ == "__main__":
    unittest.main()
