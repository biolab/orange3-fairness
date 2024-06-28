"""
This file contains the tests for the OWDatasetBias widget.
"""

import unittest

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.fairness.widgets.owdatasetbias import OWDatasetBias


class TestOWDatasetBias(WidgetTest):
    """
    Test class for the OWDatasetBias widget.
    """

    def setUp(self) -> None:
        self.data_path_adult = "https://datasets.biolab.si/core/adult.tab"
        self.incorrect_input_data_path = (
            "https://datasets.biolab.si/core/breast-cancer.tab"
        )
        self.widget = self.create_widget(OWDatasetBias)

        self.assertEqual(self.widget.disparate_impact_label.text(), "No data detected.")
        self.assertEqual(self.widget.statistical_parity_difference_label.text(), "")

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.disparate_impact_label.text(), "No data detected.")
        self.assertEqual(self.widget.statistical_parity_difference_label.text(), "")

    def test_incorrect_input_data(self):
        """
        Check that the widget displays an error message when
        the input data does not have the 'AsFairness' attributes
        """
        test_data = Table(self.incorrect_input_data_path)
        self.send_signal(self.widget.Inputs.data, test_data)
        self.assertTrue(self.widget.Error.missing_fairness_data.is_shown())

    def test_input_as_fairness_data(self):
        """Check that the widget works with data containing the fairness attributes"""
        test_data = Table(self.data_path_adult)
        self.send_signal(self.widget.Inputs.data, test_data)
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
