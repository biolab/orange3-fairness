from typing import Optional

from Orange.widgets import gui
from Orange.widgets.widget import Input, OWWidget
from Orange.data import Table

from aif360.metrics import BinaryLabelDatasetMetric

from orangedemo.utils import table_to_standard_dataset


class OWDatasetBias(OWWidget):
    name = "Dataset Bias"
    description = "Computes the bias of a dataset. More specifically, it computes the disparate impact and statistical parity difference metrics for the dataset."
    # icon = "icons/bias.svg"
    # priority = 0

    want_control_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Data", Table)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        box = gui.vBox(self.mainArea, "Bias")
        self.disparate_impact_label = gui.label(box, self, "No data detected.")
        self.statistical_parity_difference_label = gui.label(box, self, "No data detected.")

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        if (
            not data
            or not "favorable_class_value" in data.attributes
            or not "protected_attribute" in data.attributes
            or not "privileged_PA_values" in data.attributes
        ):
            return

        # Convert Orange data to aif360 StandardDataset
        standard_dataset, privileged_groups, unprivileged_groups = table_to_standard_dataset(data)

        # Compute the bias of the dataset (disparate impact and statistical parity difference)
        dataset_metric = BinaryLabelDatasetMetric(standard_dataset, unprivileged_groups, privileged_groups)
        self.disparate_impact_label.setText(f"Disparate Impact (ideal = 1): {dataset_metric.disparate_impact()}")
        self.statistical_parity_difference_label.setText(f"Statistical Parity Difference (ideal = 0): {dataset_metric.statistical_parity_difference()}")
        