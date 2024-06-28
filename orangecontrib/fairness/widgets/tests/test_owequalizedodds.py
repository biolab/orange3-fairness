"""This file contains the tests for the OWEqualizedOdds widget."""

import unittest

from Orange.widgets.tests.base import WidgetTest
from Orange.classification.logistic_regression import LogisticRegressionLearner
from Orange.widgets.evaluate.owpredictions import OWPredictions
from Orange.widgets.evaluate.owtestandscore import OWTestAndScore
from Orange.evaluation import CrossValidation, AUC, CA
from Orange.base import Model
from Orange.data import Table

from orangecontrib.fairness.evaluation import scoring as bias_scoring
from orangecontrib.fairness.widgets.owequalizedodds import OWEqualizedOdds
from orangecontrib.fairness.modeling.postprocessing import PostprocessingLearner


class TestOWEqualizedOdds(WidgetTest):
    """
    Test class for the OWEqualizedOdds widget.
    """

    def setUp(self) -> None:
        self.data_path_adult = "https://datasets.biolab.si/core/german-credit-data.tab"
        self.incorrect_input_data_path = (
            "https://datasets.biolab.si/core/breast-cancer.tab"
        )
        self.widget = self.create_widget(OWEqualizedOdds)
        self.predictions = self.create_widget(OWPredictions)
        self.test_and_score = self.create_widget(OWTestAndScore)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)

    def test_incorrect_input_data(self):
        """
        Check that the widget displays an error message when
        the input data does not have the 'AsFairness' attributes
        """
        test_data = Table(self.incorrect_input_data_path)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.assertTrue(self.widget.Error.missing_fairness_data.is_shown())

    def test_learner_output(self):
        """Check if the widget outputs a learner"""
        self.send_signal(self.widget.Inputs.input_learner, LogisticRegressionLearner())
        learner = self.widget.create_learner()

        self.assertIsNotNone(learner)

    def test_model_output(self):
        """Check if the widget outputs a model"""
        test_data = Table(self.data_path_adult)

        self.send_signal(self.widget.Inputs.input_learner, LogisticRegressionLearner())
        self.send_signal(self.widget.Inputs.data, test_data)
        self.wait_until_finished(self.widget, timeout=200000)
        model = self.get_output(self.widget.Outputs.model)

        self.assertIsNotNone(model)

    def test_compatibility_with_predictions(self):
        """Check that the widget works with the predictions widget"""
        test_data = Table(self.data_path_adult)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.send_signal(self.widget.Inputs.input_learner, LogisticRegressionLearner())
        learner = self.widget.create_learner()
        model = learner(test_data)

        self.send_signal(
            self.predictions.Inputs.data, test_data, widget=self.predictions
        )
        self.send_signal(
            self.predictions.Inputs.predictors, model, widget=self.predictions
        )
        if hasattr(self.predictions.Outputs, "predictions"):
            # OWPredictions in Orange3<3.37  has attribute named predictions
            attr = self.predictions.Outputs.predictions
        else:
            # in Orange3>=3.37 predictions is replaced with selected_predictions
            attr = self.predictions.Outputs.selected_predictions
        predictions = self.get_output(attr, widget=self.predictions)

        results = self.get_output(
            self.predictions.Outputs.evaluation_results, widget=self.predictions
        )

        self.assertIsNotNone(predictions)
        self.assertIsNotNone(results)

    def test_compatibility_with_test_and_score(self):
        """Check that the widget works with the test and score widget"""
        test_data = Table(self.data_path_adult)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.send_signal(self.widget.Inputs.input_learner, LogisticRegressionLearner())
        learner = self.widget.create_learner()

        # Change the test and score cross validation to 2 folds
        self.test_and_score.n_folds = 0

        self.send_signal(
            self.test_and_score.Inputs.train_data,
            test_data,
            widget=self.test_and_score,
        )
        self.send_signal(
            self.test_and_score.Inputs.learner, learner, widget=self.test_and_score
        )
        self.wait_until_finished(self.test_and_score, timeout=50000)
        predictions = self.get_output(
            self.test_and_score.Outputs.predictions, widget=self.test_and_score
        )
        results = self.get_output(
            self.test_and_score.Outputs.evaluations_results,
            widget=self.test_and_score,
        )

        self.assertIsNotNone(predictions)
        self.assertIsNotNone(results)

    def test_effecitveness(self):
        """Check that the widget works with the predictions widget"""
        test_data = Table(self.data_path_adult)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.send_signal(self.widget.Inputs.input_learner, LogisticRegressionLearner())
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
        self.assertNotEqual(CA(results), CA(normal_results))

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


class TestEqualizedOddsPostprocessing(unittest.TestCase):
    """
    Test class for the PostprocessingLearner and PostprocessingModel.
    """

    def setUp(self):
        self.data_path_adult = "https://datasets.biolab.si/core/adult.tab"

    def test_adversarial_learner(self):
        """Check if the adversarial learner works"""
        learner = PostprocessingLearner(LogisticRegressionLearner())
        self.assertIsNotNone(learner)
        cv = CrossValidation(k=2)
        results = cv(Table(self.data_path_adult), [learner])
        auc, ca = AUC(results), CA(results)

        self.assertGreaterEqual(auc, 0.5)
        self.assertGreaterEqual(ca, 0.5)

    def test_adversarial_model(self):
        """Check if the adversarial model works"""
        learner = PostprocessingLearner(LogisticRegressionLearner())
        data = Table(self.data_path_adult)
        model = learner(data[: len(data) // 2])
        self.assertIsNotNone(model)

        predictions = model(data[len(data) // 2 :], ret=Model.ValueProbs)
        self.assertIsNotNone(predictions)

        labels, scores = predictions

        self.assertEqual(len(labels), len(scores))
        self.assertEqual(len(labels), len(data[len(data) // 2 :]))
        self.assertLess(abs(scores.sum(axis=1) - 1).all(), 1e-6)
        self.assertTrue(all(label in [0, 1] for label in labels))


if __name__ == "__main__":
    unittest.main()
