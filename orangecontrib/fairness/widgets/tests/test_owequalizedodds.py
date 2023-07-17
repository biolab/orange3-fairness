import unittest
import os

from Orange.widgets.tests.base import WidgetTest
from Orange.classification.logistic_regression import LogisticRegressionLearner
from Orange.widgets.evaluate.owpredictions import OWPredictions
from Orange.widgets.evaluate.owtestandscore import OWTestAndScore
from Orange.evaluation import scoring

from orangecontrib.fairness.evaluation import scoring as bias_scoring
from orangecontrib.fairness.widgets.owequalizedodds import OWEqualizedOdds
from orangecontrib.fairness.widgets.owasfairness import OWAsFairness
from orangecontrib.fairness.widgets.tests.utils import as_fairness_setup


class TestOWEqualizedOdds(WidgetTest):
    def setUp(self) -> None:
        self.test_data_path = os.path.join(os.path.dirname(__file__), "datasets")
        self.widget = self.create_widget(OWEqualizedOdds)
        self.as_fairness = self.create_widget(OWAsFairness)
        self.predictions = self.create_widget(OWPredictions)
        self.cross_validation = self.create_widget(OWTestAndScore)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)

    def test_compatibility_with_predictions(self):
        """Check that the widget works with the predictions widget"""
        test_data = as_fairness_setup(self)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.send_signal(self.widget.Inputs.learner, LogisticRegressionLearner())
        learner = self.widget.create_learner()
        model = learner(test_data)

        self.send_signal(
            self.predictions.Inputs.data, test_data, widget=self.predictions
        )
        self.send_signal(
            self.predictions.Inputs.predictors, model, widget=self.predictions
        )
        predictions = self.get_output(
            self.predictions.Outputs.predictions, widget=self.predictions
        )
        results = self.get_output(
            self.predictions.Outputs.evaluation_results, widget=self.predictions
        )

        self.assertIsNotNone(predictions)
        self.assertIsNotNone(results)

    def test_compatibility_with_test_and_score(self):
        """Check that the widget works with the test and score widget"""
        test_data = as_fairness_setup(self)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.send_signal(self.widget.Inputs.learner, LogisticRegressionLearner())
        learner = self.widget.create_learner()

        self.send_signal(
            self.cross_validation.Inputs.train_data,
            test_data,
            widget=self.cross_validation,
        )
        self.send_signal(
            self.cross_validation.Inputs.learner, learner, widget=self.cross_validation
        )
        self.wait_until_finished(self.cross_validation)
        predictions = self.get_output(
            self.cross_validation.Outputs.predictions, widget=self.cross_validation
        )
        results = self.get_output(
            self.cross_validation.Outputs.evaluations_results,
            widget=self.cross_validation,
        )

        self.assertIsNotNone(predictions)
        self.assertIsNotNone(results)

    def test_effecitveness(self):
        """Check that the widget works with the predictions widget"""
        test_data = as_fairness_setup(self)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.send_signal(self.widget.Inputs.learner, LogisticRegressionLearner())
        learner = self.widget.create_learner()
        model = learner(test_data)

        # Predictions with postprocessing
        self.send_signal(
            self.predictions.Inputs.data, test_data, widget=self.predictions
        )
        self.send_signal(
            self.predictions.Inputs.predictors, model, widget=self.predictions
        )
        results = self.get_output(
            self.predictions.Outputs.evaluation_results, widget=self.predictions
        )

        # Predictions without postprocessing
        learner = LogisticRegressionLearner()
        model = learner(test_data)
        self.send_signal(
            self.predictions.Inputs.predictors, model, widget=self.predictions
        )
        normal_results = self.get_output(
            self.predictions.Outputs.evaluation_results, widget=self.predictions
        )

        # Check that the two results are different
        self.assertNotEqual(scoring.CA(results), scoring.CA(normal_results))

        aod = bias_scoring.AverageOddsDifference(results)
        normal_aod = bias_scoring.AverageOddsDifference(normal_results)
        self.assertNotEqual(aod, normal_aod)

        # Check that the absolute value of aod is smaller than the normal aod
        # self.assertLessEqual(np.abs(aod), np.abs(normal_aod))

    def test_repeatable_parameter(self):
        """Check that the repeatable parameter works"""
        self.widget.repeatable = True
        self.assertTrue(self.widget.repeatable)

        self.widget.repeatable = False
        self.assertFalse(self.widget.repeatable)


if __name__ == "__main__":
    unittest.main()
