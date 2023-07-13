import os
import unittest

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_rows

from orangedemo.owasfairness import OWAsFairness
from orangedemo.owdatasetbias import OWDatasetBias


class TestOWDatasetBias(WidgetTest):
    def setUp(self) -> None:
        self.test_data_path = os.path.join(os.path.dirname(__file__), "datasets")
        self.widget = self.create_widget(OWDatasetBias)
        self.as_fairness = self.create_widget(OWAsFairness)

        self.assertEqual(self.widget.disparate_impact_label.text(), "No data detected.")
        self.assertEqual(self.widget.statistical_parity_difference_label.text(), "")

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.disparate_impact_label.text(), "No data detected.")
        self.assertEqual(self.widget.statistical_parity_difference_label.text(), "")

    def test_input_normal_data(self):
        """Check that the widget doesn't crash on normal data"""
        test_data = Table(f"{self.test_data_path}/adult.tab")
        self.send_signal(self.widget.Inputs.data, test_data)
        self.assertEqual(
            self.widget.disparate_impact_label.text(),
            "The dataset is not suitable for bias computation.",
        )
        self.assertEqual(
            self.widget.statistical_parity_difference_label.text(),
            "Pass the dataset through the 'As Fairness' widget first",
        )

    def test_input_as_fairness_data(self):
        """Check that the widget works with data from the as fairness widget"""
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

        self.send_signal(self.widget.Inputs.data, output_data)
        self.assertTrue(
            self.widget.disparate_impact_label.text().startswith(
                "Disparate Impact (ideal = 1):"
            )
        )
        self.assertTrue(
            self.widget.statistical_parity_difference_label.text().startswith(
                "Statistical Parity Difference (ideal = 0):"
            )
        )


if __name__ == "__main__":
    unittest.main()
