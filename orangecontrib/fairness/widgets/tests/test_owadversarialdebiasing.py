import unittest

from Orange.evaluation import CrossValidation, AUC, CA
from Orange.base import Model
from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table

from orangecontrib.fairness.widgets.owadversarialdebiasing import OWAdversarialDebiasing
from orangecontrib.fairness.modeling.adversarial import AdversarialDebiasingLearner

class TestOWAdversarialDebiasing(WidgetTest):
    def setUp(self):
        self.data_path_adult = "https://datasets.biolab.si/core/adult.tab"
        self.incorrect_input_data_path = "https://datasets.biolab.si/core/breast-cancer.tab"
        self.widget = self.create_widget(OWAdversarialDebiasing)

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

    def test_incorrect_input_data(self):
        """Check that the widget displays an error message when the input data does not have the 'AsFairness' attributes"""
        test_data = Table(self.incorrect_input_data_path)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.assertTrue(self.widget.Error.missing_fairness_data.is_shown())

    def test_learner_output(self):
        """Check if the widget outputs a learner"""
        learner = self.widget.create_learner()

        self.assertIsNotNone(learner)

    def test_model_output(self):
        """Check if the widget outputs a model"""
        self.widget.number_of_epochs = 5
        self.widget.debias = True
        test_data = Table(self.data_path_adult)

        self.send_signal(self.widget.Inputs.data, test_data)
        self.wait_until_finished(self.widget, timeout=200000)
        model = self.get_output(self.widget.Outputs.model)

        self.assertIsNotNone(model)


class TestAdversarialDebiasing(unittest.TestCase):
    def setUp(self):
        # self.data_path_adult = "https://datasets.biolab.si/core/adult.tab"
        # self.data_path_adult = "https://datasets.biolab.si/core/compas-scores-two-years.tab"
        self.data_path_adult = "https://datasets.biolab.si/core/german-credit-data.tab"

    def test_adversarial_learner(self):
        """Check if the adversarial learner works"""
        learner = AdversarialDebiasingLearner(num_epochs=20)
        self.assertIsNotNone(learner)
        cv = CrossValidation(k=2)
        results = cv(Table(self.data_path_adult), [learner])
        auc, ca = AUC(results), CA(results)

        self.assertGreaterEqual(auc, 0.5)
        self.assertGreaterEqual(ca, 0.5)

    def test_adversarial_model(self):
        """Check if the adversarial model works"""
        learner = AdversarialDebiasingLearner(num_epochs=20, seed=42)
        data = Table(self.data_path_adult)
        model = learner(data[:len(data) // 2])
        self.assertIsNotNone(model)

        predictions = model(data[len(data) // 2:], ret=Model.ValueProbs )
        self.assertIsNotNone(predictions)

        labels, scores = predictions

        self.assertEqual(len(labels), len(scores))
        self.assertEqual(len(labels), len(data[len(data) // 2:]))
        self.assertLess(abs(scores.sum(axis=1) - 1).all(), 1e-6)
        self.assertTrue(all(label in [0, 1] for label in labels))


# class TestCallbackSession(unittest.TestCase):
#     """
#     In the adversarial.py file create a Subclass of tensorflow session with callback functionality for progress tracking and displaying.
#     This class should be tested to ensure that the tf.Session has not been modified in a way that breaks the functionality of the widget.
#     """

#     def setUp(self):
#         self.data_path_adult = "https://datasets.biolab.si/core/adult.tab"
#         self.data = Table(self.data_path_adult)
#         self.run_count = 0
#         self.last_received_progress = None

#     def callback_function(self, progress, msg=""):
#         """Callback function that increments the run count and stores the received progress."""
#         self.run_count += 1
#         self.last_received_progress = progress

#     def test_callback_with_learner(self):
#         # Define the learner
#         learner = AdversarialDebiasingLearner(num_epochs=20, batch_size=128)
#         expected_total_runs = learner._calculate_total_runs(self.data)

#         # Fit the learner to the data with the test callback function
#         learner(self.data, progress_callback=self.callback_function)

#         # Validate callback was called correct number of times (+- 15 runs)
#         self.assertAlmostEqual(self.run_count, expected_total_runs, delta=15)

#         # Validate the progress callback values. It should be between 0 to 100.
#         self.assertTrue(0 <= self.last_received_progress <= 100)




if __name__ == "__main__":
    unittest.main()











# def test_cross_validation(self):
#     """Check if the widget works with cross validation"""
#     self.widget.number_of_epochs = 10
#     self.widget.debias = False

#     test_data = Table(self.data_path_adult)

#     learner = self.widget.create_learner()

#     cv = CrossValidation(k=5, random_state=42, store_data=True)
#     results = cv(test_data, [learner])

#     self.assertIsNotNone(results)
#     print("Cross validation results:")
#     print_metrics(results)

# def test_train_test_split(self):
#     """Check if the widget works with a normal train-test split"""
#     self.widget.number_of_epochs = 10
#     self.widget.debias = False

#     test_data = Table(self.data_path_adult)

#     learner = self.widget.create_learner()

#     test_on_training = TestOnTrainingData(store_data=True)
#     results = test_on_training(test_data, [learner])

#     self.assertIsNotNone(results)
#     print("Train test split results:")
#     print_metrics(results)


# def test_compatibility_with_test_and_score(self):
#     """Check that the widget works with the predictions widget"""
#     self.test_and_score = self.create_widget(OWTestAndScore)
    
#     self.widget.number_of_epochs = 10
#     self.widget.debias = False

#     data_sample = Table("orangedemo/tests/datasets/adult_sample.pkl")
#     data_remaining = Table("orangedemo/tests/datasets/adult_remaining.pkl")
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

#     data_sample = Table("orangedemo/tests/datasets/adult_sample.pkl")
#     data_remaining = Table("orangedemo/tests/datasets/adult_remaining.pkl")
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