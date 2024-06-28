"""
This module contains the implementation of the Dataset Bias widget.

This widget computes the bias of a dataset. More specifically, it computes 
the disparate impact and statistical parity difference metrics for the dataset.
"""

from typing import Optional

from Orange.widgets import gui
from Orange.widgets.widget import Input, OWWidget
from Orange.data import Table

from aif360.metrics import BinaryLabelDatasetMetric

from orangecontrib.fairness.widgets.utils import (
    table_to_standard_dataset,
    check_fairness_data,
    check_for_missing_values,
)


class OWDatasetBias(OWWidget):
    """
    Widget for computing the fairness metrics (bias) of a dataset.
    More specifically, it computes the disparate impact and statistical
    parity difference metrics for the dataset.
    """

    name = "Dataset Bias"
    description = (
        "Computes the bias of a dataset. More specifically, it computes the disparate "
        "impact and statistical parity difference metrics for the dataset."
    )
    icon = "icons/dataset_bias.svg"
    priority = 10

    want_control_area = False
    resizing_enabled = False

    class Inputs:
        """Input for the widget - dataset."""

        data = Input("Data", Table)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        box = gui.vBox(self.mainArea, "Bias")
        self.disparate_impact_label = gui.label(box, self, "No data detected.")
        self.statistical_parity_difference_label = gui.label(box, self, "")

    @Inputs.data
    @check_fairness_data
    @check_for_missing_values
    def set_data(self, data: Optional[Table]) -> None:
        """Computes the bias of the dataset and displays it on the widget."""
        if not data:
            self.disparate_impact_label.setText("No data detected.")
            self.disparate_impact_label.setToolTip("")
            self.statistical_parity_difference_label.setText("")
            self.statistical_parity_difference_label.setToolTip("")
            return

        # Convert Orange data to aif360 StandardDataset
        standard_dataset, privileged_groups, unprivileged_groups = (
            table_to_standard_dataset(data)
        )

        # Compute the bias of the dataset (disparate impact and statistical parity difference)
        dataset_metric = BinaryLabelDatasetMetric(
            standard_dataset, unprivileged_groups, privileged_groups
        )
        disparate_impact = dataset_metric.disparate_impact()
        statistical_parity_difference = dataset_metric.statistical_parity_difference()
        self.disparate_impact_label.setText(
            f"Disparate Impact (ideal = 1): {round(disparate_impact, 3)}"
        )
        self.disparate_impact_label.setToolTip(
            "<p>Disparate Impact (DI): Measures the ratio of the ratios of favorable class "
            "values for an unprivileged group to that of the privileged group. An ideal value "
            "of 1.0 means the ratio of favorable class values is the same for both groups.</p>"
            "<ul>"
            "<li>DI &lt; 1.0: The privileged group has a higher percentage of favorable class values.</li>"
            "<li>DI &gt; 1.0: The privileged group has a lower percentage of favorable class values.</li>"
            "</ul>"
        )
        self.statistical_parity_difference_label.setText(
            f"Statistical Parity Difference (ideal = 0): {round(statistical_parity_difference, 3)}"
        )
        self.statistical_parity_difference_label.setToolTip(
            "<p>Statistical Parity Difference (SPD): Measures the difference in ratios of "
            "favorable class values between the unprivileged and the privileged groups. An "
            "ideal value for this metric is 0.</p>"
            "<ul>"
            "<li>SPD &lt; 0: The privileged group has a higher percentage of favorable class values.</li>"
            "<li>SPD &gt; 0: The privileged group has a lower percentage of favorable class values.</li>"
            "</ul>"
        )
