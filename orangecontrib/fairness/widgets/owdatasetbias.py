from typing import Optional

from Orange.widgets import gui
from Orange.widgets.widget import Input, OWWidget
from Orange.data import Table

from aif360.metrics import BinaryLabelDatasetMetric

from orangecontrib.fairness.widgets.utils import table_to_standard_dataset, check_fairness_data


class OWDatasetBias(OWWidget):
    name = "Dataset Bias"
    description = "Computes the bias of a dataset. More specifically, it computes the disparate impact and statistical parity difference metrics for the dataset."
    icon = "icons/dataset_bias.svg"
    # priority = 0

    want_control_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Data", Table)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        box = gui.vBox(self.mainArea, "Bias")
        self.disparate_impact_label = gui.label(box, self, "No data detected.")
        self.statistical_parity_difference_label = gui.label(box, self, "")

    @Inputs.data
    @check_fairness_data
    def set_data(self, data: Optional[Table]) -> None:
        if (
            not data
        ):
            self.disparate_impact_label.setText("No data detected.")
            self.statistical_parity_difference_label.setText("")
            return

        # Convert Orange data to aif360 StandardDataset
        standard_dataset, privileged_groups, unprivileged_groups = table_to_standard_dataset(data)

        # Compute the bias of the dataset (disparate impact and statistical parity difference)
        dataset_metric = BinaryLabelDatasetMetric(standard_dataset, unprivileged_groups, privileged_groups)
        disparate_impact = dataset_metric.disparate_impact()
        statistical_parity_difference = dataset_metric.statistical_parity_difference()
        self.disparate_impact_label.setText(f"Disparate Impact (ideal = 1): {round(disparate_impact, 3)}")
        self.statistical_parity_difference_label.setText(f"Statistical Parity Difference (ideal = 0): {round(statistical_parity_difference, 3)}")
        