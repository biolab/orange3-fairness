import os
import unittest

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_rows

from orangedemo.owasfairness import OWAsFairness
from orangedemo.owreweighing import OWReweighing


class TestOWReweighing(WidgetTest):
    def setUp(self) -> None:
        self.test_data_path = os.path.join(os.path.dirname(__file__), "datasets")
        self.widget = self.create_widget(OWReweighing)
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
        simulate.combobox_activate_item(
            self.as_fairness.controls.protected_attribute, "sex"
        )
        select_rows(self.as_fairness.controls.privileged_PA_values, [1])
        output_data = self.get_output(self.as_fairness.Outputs.data)
        return output_data

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNone(self.get_output(self.widget.Outputs.preprocessor))

    def test_output_data(self):
        """Check that the widget handles data correctly and adds the 'weights' column"""
        test_data = self.as_fairness_setup()
        self.send_signal(self.widget.Inputs.data, test_data)
        output_data = self.get_output(self.widget.Outputs.data)
        self.assertIsNotNone(output_data)
        self.assertIn("weights", output_data.domain)

    def test_preprocessor_output(self):
        """Check that the widget returns a working preprocessor"""
        test_data = self.as_fairness_setup()
        self.send_signal(self.widget.Inputs.data, test_data)
        preprocessor = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsNotNone(preprocessor)
        preprocessed_data = preprocessor(test_data)
        self.assertIsNotNone(preprocessed_data)
        self.assertIn("weights", preprocessed_data.domain)


if __name__ == "__main__":
    unittest.main()
