import os
import unittest

from Orange.evaluation import CrossValidation, TestOnTrainingData
from Orange.widgets.tests.base import WidgetTest

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
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

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
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        learner = self.widget.create_learner()

        ttt = TestOnTrainingData(store_data=True)
        results = ttt(test_data, [learner])

        self.assertIsNotNone(results)
        print("Train test split results:")
        print_metrics(results)


if __name__ == "__main__":
    unittest.main()
