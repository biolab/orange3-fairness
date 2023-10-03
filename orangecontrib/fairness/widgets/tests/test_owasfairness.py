import unittest

from Orange.data.table import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_rows

from orangecontrib.fairness.widgets.owasfairness import OWAsFairness
from orangecontrib.fairness.widgets.tests.utils import fairness_attributes



class TestOWAsFairness(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWAsFairness)
        self.data_path_adult = "https://datasets.biolab.si/core/adult.tab"
        # self.data_path_compas = "https://datasets.biolab.si/core/compas-scores-two-years.tab"
        # self.data_path_german = "https://datasets.biolab.si/core/german-credit-data.tab"

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, None)

    def test_input_data(self):
        """Check that the comboboxes are populated with the values"""
        test_data = Table(self.data_path_adult)
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        self.assertTrue(self.widget.controls.protected_attribute.count() > 0)
        self.assertTrue(self.widget.controls.favorable_class_value.count() > 0)
        self.assertTrue(
            self.widget.controls.privileged_pa_values.model().rowCount() > 0
        )

    def test_display_default(self):
        """Check that the widget automatically displays the default fairness attributes if the input data contains them"""
        test_data = Table(self.data_path_adult)
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        favorable_class_value, protected_attribute, privileged_pa_values = fairness_attributes(test_data.domain)

        self.assertEqual(self.widget.controls.favorable_class_value.currentText(), favorable_class_value)
        
        self.assertEqual(self.widget.controls.protected_attribute.currentText(), protected_attribute.name)

        selected_indexes = self.widget.controls.privileged_pa_values.selectionModel().selectedRows()
        model = self.widget.controls.privileged_pa_values.model()
        selected_values = [model.data(index) for index in selected_indexes]
        self.assertEqual(selected_values, privileged_pa_values)

    def test_select_default(self):
        """Check that the widget automatically selects the default fairness attributes if the input data contains them"""
        test_data = Table(self.data_path_adult)
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        favorable_class_value, protected_attribute, privileged_pa_values = fairness_attributes(test_data.domain)
        self.assertEqual(self.widget.favorable_class_value, favorable_class_value)
        self.assertEqual(self.widget.protected_attribute.name, protected_attribute.name)
        self.assertEqual(self.widget.privileged_pa_values, privileged_pa_values)


    def test_selection(self):
        """Check that the selection of fairness attributes works properly"""
        test_data = Table(self.data_path_adult)
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        # Test that the selection of favorable class value works
        simulate.combobox_activate_index(self.widget.controls.favorable_class_value, 0)
        self.assertEqual(self.widget.favorable_class_value, self.widget.controls.favorable_class_value.currentText())

        # Test that the selection of protected attribute works
        simulate.combobox_activate_index(self.widget.controls.protected_attribute, 0)
        self.assertEqual(self.widget.protected_attribute.name, self.widget.controls.protected_attribute.currentText())

        # Test that the selection of privileged protected attribute values works
        select_rows(self.widget.controls.privileged_pa_values, [1])
        selected_indexes = self.widget.controls.privileged_pa_values.selectionModel().selectedRows()

        model = self.widget.controls.privileged_pa_values.model()
        selected_values = [model.data(index) for index in selected_indexes]

        self.assertEqual(self.widget.privileged_pa_values, selected_values)

    def test_output(self):
        """Check that the selection of fairness attributes properly modifies the output data"""
        test_data = Table(self.data_path_adult)
        self.send_signal(
            self.widget.Inputs.data,
            test_data,
        )

        # Select the fairness attributes
        simulate.combobox_activate_index(self.widget.controls.favorable_class_value, 0)
        simulate.combobox_activate_index(self.widget.controls.protected_attribute, 0)
        select_rows(self.widget.controls.privileged_pa_values, [1])

        # Check that the output data contains the fairness attributes
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
