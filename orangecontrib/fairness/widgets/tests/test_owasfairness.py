import os
import unittest

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_rows

from orangecontrib.fairness.widgets.owasfairness import OWAsFairness


class TestOWAsFairness(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWAsFairness)
        self.test_data_path = "https://datasets.biolab.si/core/adult.tab"

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)

    def test_input_data(self):
        """Check that the comboboxes are populated with the values"""
        test_data = Table(self.test_data_path)
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        self.assertTrue(self.widget.controls.protected_attribute.count() > 0)
        self.assertTrue(self.widget.controls.favorable_class_value.count() > 0)
        self.assertTrue(
            self.widget.controls.privileged_pa_values.model().rowCount() > 0
        )

    def test_selection(self):
        """Check that the selection works properly"""
        test_data = Table(self.test_data_path)
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        simulate.combobox_activate_item(
            self.widget.controls.favorable_class_value, ">50K"
        )
        self.assertEqual(self.widget.favorable_class_value, ">50K")

        simulate.combobox_activate_item(self.widget.controls.protected_attribute, "sex")
        self.assertEqual(self.widget.protected_attribute.name, "sex")

        select_rows(self.widget.controls.privileged_pa_values, [1])
        self.assertEqual(self.widget.privileged_pa_values, ["Male"])

    def test_output(self):
        """Check that the selection is properly set"""
        test_data = Table(self.test_data_path)
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        simulate.combobox_activate_item(
            self.widget.controls.favorable_class_value, ">50K"
        )
        simulate.combobox_activate_item(self.widget.controls.protected_attribute, "sex")
        select_rows(self.widget.controls.privileged_pa_values, [1])

        output_data = self.get_output(self.widget.Outputs.data)

        self.assertTrue(output_data is not None)
        self.assertTrue("favorable_class_value" in output_data.domain.class_var.attributes)
        contains_pa_values = False
        for attr in output_data.domain.attributes:
            if "privileged_pa_values" in attr.attributes:
                        contains_pa_values = True
        self.assertTrue(contains_pa_values)


if __name__ == "__main__":
    unittest.main()
