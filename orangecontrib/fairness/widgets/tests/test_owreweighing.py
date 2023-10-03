import unittest
import numpy as np

from Orange.widgets.tests.base import WidgetTest
from Orange.preprocess.preprocess import PreprocessorList
from Orange.data import Table

from orangecontrib.fairness.widgets.owreweighing import OWReweighing
from orangecontrib.fairness.widgets.owweightedlogisticregression import OWWeightedLogisticRegression
from orangecontrib.fairness.widgets.owcombinepreprocessors import OWCombinePreprocessors



class TestOWReweighing(WidgetTest):
    def setUp(self) -> None:
        self.data_path_adult = "https://datasets.biolab.si/core/compas-scores-two-years.tab"
        self.incorrect_input_data_path = "https://datasets.biolab.si/core/breast-cancer.tab"
        self.widget = self.create_widget(OWReweighing)
        self.combine_preprocessors = self.create_widget(OWCombinePreprocessors)
        self.weighted_logistic_regression = self.create_widget(OWWeightedLogisticRegression)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.preprocessor))

    def test_incorrect_input_data(self):
        """Check that the widget displays an error message when the input data does not have the fairness attributes"""
        test_data = Table(self.incorrect_input_data_path)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.assertTrue(self.widget.Error.missing_fairness_data.is_shown())

    def test_output_data(self):
        """Check that the widget handles data correctly and adds the 'weights' column"""
        test_data = Table(self.data_path_adult)
        self.send_signal(self.widget.Inputs.data, test_data)
        output_data = self.get_output(self.widget.Outputs.data)
        self.assertIsNotNone(output_data)
        self.assertIn("weights", output_data.domain)

    def test_preprocessor_output(self):
        """Check that the widget returns a working preprocessor"""
        test_data = Table(self.data_path_adult)
        preprocessor = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsNotNone(preprocessor)
        preprocessed_data = preprocessor(test_data)
        self.assertIsNotNone(preprocessed_data)
        self.assertIn("weights", preprocessed_data.domain)

    def test_combine_preprocessors(self):
        """Check that the combine preprocessors widget works correctly"""
        first_preprocessor = self.get_output(self.widget.Outputs.preprocessor)
        second_preprocessor = self.get_output(self.widget.Outputs.preprocessor)

        self.send_signal(self.combine_preprocessors.Inputs.first_preprocessor, first_preprocessor)
        self.send_signal(self.combine_preprocessors.Inputs.second_preprocessor, second_preprocessor)

        combined_preprocessor = self.get_output(self.combine_preprocessors.Outputs.preprocessor)

        # Check that the output is not None
        self.assertIsNotNone(combined_preprocessor)

        # Check that the output is of type PreprocessorList
        self.assertEqual(type(combined_preprocessor), PreprocessorList)

        # Check that there are two preprocessors in the list
        self.assertEqual(len(combined_preprocessor.preprocessors), 2)


    def test_with_weighted_logistic_regression(self):
        """Check that the predictions of logistic regression on the original data and the preprocessed data are different"""

        test_data = Table(self.data_path_adult)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.wait_until_finished(self.widget)
        preprocessed_data = self.get_output(self.widget.Outputs.data)
        self.assertIsNotNone(preprocessed_data)

        # Train a model on the original data
        self.send_signal(self.weighted_logistic_regression.Inputs.data, test_data)
        self.wait_until_finished(self.weighted_logistic_regression)
        normal_model = self.get_output(self.weighted_logistic_regression.Outputs.model)

        # Train a model on the preprocessed data
        self.send_signal(self.weighted_logistic_regression.Inputs.data, preprocessed_data)
        self.wait_until_finished(self.weighted_logistic_regression)
        preprocessed_model = self.get_output(self.weighted_logistic_regression.Outputs.model)

        # Check that the predictions of the two models are different
        self.assertFalse(
            np.array_equal(
                normal_model(test_data),
                preprocessed_model(preprocessed_data)
            ),
            "Preprocessed predictions should not equal normal predictions",
        )


if __name__ == "__main__":
    unittest.main()
