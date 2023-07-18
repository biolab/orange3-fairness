import os
import unittest

from Orange.evaluation import CrossValidation, TestOnTrainingData, TestOnTestData
from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table
from Orange.widgets.evaluate.owpredictions import OWPredictions
from Orange.widgets.evaluate.owtestandscore import OWTestAndScore

from orangecontrib.fairness.widgets.tests.utils import as_fairness_setup, print_metrics
from orangecontrib.fairness.widgets.owasfairness import OWAsFairness
from orangecontrib.fairness.widgets.owadversarialdebiasing import OWAdversarialDebiasing


class TestOWAdversarialDebiasing(WidgetTest):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(__file__), "datasets")
        self.widget = self.create_widget(OWAdversarialDebiasing)
        self.as_fairness = self.create_widget(OWAsFairness)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)

    def test_parameters(self):
        """Check the selection of parameters"""
        # Change settings
        self.widget.hidden_layers_neurons = 50
        self.widget.number_of_epochs = 100
        self.widget.batch_size = 64
        self.widget.debias = False
        self.widget.repeatable = True

        # Check that settings have changed
        self.assertEqual(self.widget.hidden_layers_neurons, 50)
        self.assertEqual(self.widget.number_of_epochs, 100)
        self.assertEqual(self.widget.batch_size, 64)
        self.assertEqual(self.widget.debias, False)
        self.assertEqual(self.widget.repeatable, True)

    def test_cross_validation(self):
        """Check if the widget works with cross validation"""
        self.widget.number_of_epochs = 10
        self.widget.debias = False

        test_data = as_fairness_setup(self)

        learner = self.widget.create_learner()

        cv = CrossValidation(k=5, random_state=42, store_data=True)
        results = cv(test_data, [learner])

        self.assertIsNotNone(results)
        print("Cross validation results:")
        print_metrics(results)

    def test_train_test_split(self):
        """Check if the widget works with a normal train-test split"""
        self.widget.number_of_epochs = 10
        self.widget.debias = False

        test_data = as_fairness_setup(self)

        learner = self.widget.create_learner()

        test_on_training = TestOnTrainingData(store_data=True)
        results = test_on_training(test_data, [learner])

        self.assertIsNotNone(results)
        print("Train test split results:")
        print_metrics(results)

    # def test_compatibility_with_test_and_score(self):
    #     """Check that the widget works with the predictions widget"""
    #     self.test_and_score = self.create_widget(OWTestAndScore)
        
    #     self.widget.number_of_epochs = 10
    #     self.widget.debias = False

    #     data_sample = Table("workflows/testing_data/adult_sample.pkl")
    #     data_remaining = Table("workflows/testing_data/adult_remaining.pkl")
    #     self.send_signal(self.widget.Inputs.data, data_sample)

    #     self.wait_until_finished(self.widget, timeout=2000000)

    #     learner = self.get_output(self.widget.Outputs.learner)

    #     self.send_signal(
    #         self.test_and_score.Inputs.train_data, data_remaining, widget=self.test_and_score
    #     )
    #     self.send_signal(
    #         self.test_and_score.Inputs.learner, learner, widget=self.test_and_score
    #     )
    #     results = self.get_output(
    #         self.test_and_score.Outputs.evaluations_results, widget=self.test_and_score
    #     )

    #     print_metrics(results)

    # def test_compatibility_with_predictions(self):
    #     """Check that the widget works with the predictions widget"""
    #     self.predictions = self.create_widget(OWPredictions)
        
    #     self.widget.number_of_epochs = 10
    #     self.widget.debias = False

    #     data_sample = Table("workflows/testing_data/adult_sample.pkl")
    #     data_remaining = Table("workflows/testing_data/adult_remaining.pkl")
    #     self.send_signal(self.widget.Inputs.data, data_sample)

    #     self.wait_until_finished(self.widget, timeout=2000000)

    #     model = self.get_output(self.widget.Outputs.model)

    #     self.send_signal(
    #         self.predictions.Inputs.data, data_remaining, widget=self.predictions
    #     )
    #     self.send_signal(
    #         self.predictions.Inputs.predictors, model, widget=self.predictions
    #     )
    #     results = self.get_output(
    #         self.predictions.Outputs.evaluation_results, widget=self.predictions
    #     )

    #     print_metrics(results)

    # def test_try_to_replicate_error(self):
    #     """Check if the widget works with a normal train-test split"""
    #     self.widget.number_of_epochs = 10
    #     self.widget.debias = False

    #     data_sample = Table("workflows/testing_data/adult_sample.pkl")
    #     data_remaining = Table("workflows/testing_data/adult_remaining.pkl")

    #     learner = self.widget.create_learner()

    #     test_on_test = TestOnTestData(store_data=True)
    #     results = test_on_test(data=data_sample, test_data=data_remaining, learners=[learner])

    #     self.assertIsNotNone(results)
    #     print("Train test split results:")
    #     print_metrics(results)


if __name__ == "__main__":
    unittest.main()
