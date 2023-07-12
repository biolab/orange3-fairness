import os
import unittest

from Orange.data.table import Table
from Orange.evaluation import CrossValidation, TestOnTrainingData, scoring
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_rows

from orangedemo.owasfairness import OWAsFairness
from orangedemo.owadversarialdebiasing import OWAdversarialDebiasing
from orangedemo.evaluation import scoring as bias_scoring


class TestOWAdversarialDebiasing(WidgetTest):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(__file__), "datasets")
        self.widget = self.create_widget(OWAdversarialDebiasing)
        self.as_fairness = self.create_widget(OWAsFairness)

    def as_fairness_setup(self):
        test_data = Table(f"{self.test_data_path}/adult.tab")
        self.send_signal(
            self.as_fairness.Inputs.data,
            test_data,
        )
        simulate.combobox_activate_item(
            self.as_fairness.controls.favorable_class_value, ">50K"
        )
        simulate.combobox_activate_item(self.as_fairness.controls.protected_attribute, "sex")
        select_rows(self.as_fairness.controls.privileged_PA_values, [1])
        output_data = self.get_output(self.as_fairness.Outputs.data)
        return output_data

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
        
        test_data = self.as_fairness_setup()
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        learner = self.widget.create_learner()

        cv = CrossValidation(k=5, random_state=42, store_data=True)
        results = cv(test_data, [learner])

        self.assertIsNotNone(results)
        print("Cross validation results:")
        print(f"ROC AUC: {scoring.AUC(results)}")
        print(f"CA: {scoring.CA(results)}")
        print(f"F1: {scoring.F1(results)}")
        print(f"Precision: {scoring.Precision(results)}")
        print(f"Recall: {scoring.Recall(results)}")
        print(f"SPD: {bias_scoring.StatisticalParityDifference(results)}")
        print(f"EOD: {bias_scoring.EqualOpportunityDifference(results)}")
        print(f"AOD: {bias_scoring.AverageOddsDifference(results)}")
        print(f"DI: {bias_scoring.DisparateImpact(results)}")

    def test_train_test_split(self):
        """Check if the widget works with a normal train-test split"""
        self.widget.number_of_epochs = 10
        self.widget.debias = False

        test_data = self.as_fairness_setup()
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        learner = self.widget.create_learner()

        ttt = TestOnTrainingData(store_data=True)
        results = ttt(test_data, [learner])

        self.assertIsNotNone(results)
        print("Train test split results:")
        print(f"ROC AUC: {scoring.AUC(results)}")
        print(f"CA: {scoring.CA(results)}")
        print(f"F1: {scoring.F1(results)}")
        print(f"Precision: {scoring.Precision(results)}")
        print(f"Recall: {scoring.Recall(results)}")
        print(f"SPD: {bias_scoring.StatisticalParityDifference(results)}")
        print(f"EOD: {bias_scoring.EqualOpportunityDifference(results)}")
        print(f"AOD: {bias_scoring.AverageOddsDifference(results)}")
        print(f"DI: {bias_scoring.DisparateImpact(results)}")


if __name__ == "__main__":
    unittest.main()