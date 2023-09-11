import unittest

from Orange.widgets.tests.base import WidgetTest
from Orange.classification.logistic_regression import LogisticRegressionLearner
from Orange.widgets.evaluate.owpredictions import OWPredictions
from Orange.widgets.evaluate.owtestandscore import OWTestAndScore
from Orange.evaluation import scoring
from Orange.data import Table

from orangecontrib.fairness.evaluation import scoring as bias_scoring
from orangecontrib.fairness.widgets.owequalizedodds import OWEqualizedOdds
from orangecontrib.fairness.widgets.owasfairness import OWAsFairness
from orangecontrib.fairness.widgets.tests.utils import as_fairness_setup


class TestOWEqualizedOdds(WidgetTest):
    def setUp(self) -> None:
        self.test_data_path = "https://datasets.biolab.si/core/adult.tab"
        self.test_incorrect_input_data_path = "https://datasets.biolab.si/core/breast-cancer.tab"
        self.widget = self.create_widget(OWEqualizedOdds)
        self.as_fairness = self.create_widget(OWAsFairness)
        self.predictions = self.create_widget(OWPredictions)
        self.test_and_score = self.create_widget(OWTestAndScore)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)

    def test_incorrect_input_data(self):
        """Check that the widget displays an error message when the input data does not have the 'AsFairness' attributes"""
        test_data = Table(self.test_incorrect_input_data_path)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.assertTrue(self.widget.Error.missing_fairness_data.is_shown())

    def test_compatibility_with_predictions(self):
        """Check that the widget works with the predictions widget"""
        test_data = as_fairness_setup(self)
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
        self.wait_until_finished(self.test_and_score, timeout=20000)
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
        test_data = as_fairness_setup(self)
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


    # def test_with_learner_widget(self):
    #     from Orange.widgets.model.owrandomforest import OWRandomForest

    #     random_forest_widget = self.create_widget(OWRandomForest)
    #     learner = self.get_output(random_forest_widget.Outputs.learner, random_forest_widget)

    #     self.assertIsNotNone(learner)

    #     test_data = as_fairness_setup(self)
    #     self.send_signal(self.widget.Inputs.data, test_data)
    #     self.send_signal(self.widget.Inputs.learner, learner)
    #     learner = self.widget.create_learner()
    #     model = learner(test_data)

    #     self.send_signal(
    #         self.predictions.Inputs.data, test_data, widget=self.predictions
    #     )
    #     self.send_signal(
    #         self.predictions.Inputs.predictors, model, widget=self.predictions
    #     )
    #     predictions = self.get_output(
    #         self.predictions.Outputs.predictions, widget=self.predictions
    #     )
    #     results = self.get_output(
    #         self.predictions.Outputs.evaluation_results, widget=self.predictions
    #     )

    #     self.assertIsNotNone(predictions)
    #     self.assertIsNotNone(results)


if __name__ == "__main__":
    unittest.main()
