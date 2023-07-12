import os
import unittest

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_rows

from orangedemo.owasfairness import OWAsFairness


class TestOWAsFairness(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWAsFairness)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)

    def test_input_data(self):
        """Check that the comboboxes are populated with the values"""
        test_data_path = os.path.join(os.path.dirname(__file__), "datasets")
        test_data = Table(f"{test_data_path}/adult.tab")
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        self.assertTrue(self.widget.controls.protected_attribute.count() > 0)
        self.assertTrue(self.widget.controls.favorable_class_value.count() > 0)
        self.assertTrue(
            self.widget.controls.privileged_PA_values.model().rowCount() > 0
        )

    def test_selection(self):
        """Check that the selection works properly"""
        test_data_path = os.path.join(os.path.dirname(__file__), "datasets")
        test_data = Table(f"{test_data_path}/adult.tab")
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

        select_rows(self.widget.controls.privileged_PA_values, [1])
        self.assertEqual(self.widget.privileged_PA_values, ["Male"])

    def test_output(self):
        """Check that the selection is properly set"""
        test_data_path = os.path.join(os.path.dirname(__file__), "datasets")
        test_data = Table(f"{test_data_path}/adult.tab")
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        simulate.combobox_activate_item(
            self.widget.controls.favorable_class_value, ">50K"
        )
        simulate.combobox_activate_item(self.widget.controls.protected_attribute, "sex")
        select_rows(self.widget.controls.privileged_PA_values, [1])

        output_data = self.get_output(self.widget.Outputs.data)

        self.assertTrue(output_data is not None)
        self.assertTrue("favorable_class_value" in output_data.attributes)
        self.assertTrue("protected_attribute" in output_data.attributes)
        self.assertTrue("privileged_PA_values" in output_data.attributes)


if __name__ == "__main__":
    unittest.main()
