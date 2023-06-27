from typing import Optional

from Orange.widgets import gui
from Orange.widgets.widget import Input, OWWidget
from Orange.widgets import gui
from Orange.widgets.widget import Input, OWWidget
from Orange.data import Table

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric

import sys

class OWDatasetBias(OWWidget):
    name = "Dataset Bias"
    description = "Computes the bias of a dataset. More specifically, it computes the disparate impact and statistical parity difference metrics for the dataset."
    # icon = "icons/bias.svg"
    # priority = 0

    want_control_area = False
    resizing_enabled = False

    standardDataset = None
    unprivileged_groups = None
    privileged_groups = None
    
    class Inputs:
        data = Input("Data", Table)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.disparate_impact = 0.0
        self.statistical_parity_difference = 0.0
        self._data: Optional[Table] = None

        box = gui.vBox(self.mainArea, "Bias")
        self.DI_label = gui.label(box, self, "No data detected.")
        self.SPD_label = gui.label(box, self, "No data detected.")

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        if not data or not "favorable_class_value" in data.attributes or not "protected_attribute" in data.attributes or not "privileged_PA_values" in data.attributes:
            return
        
        self._data = data

        # Convert Orange data to aif360 StandardDataset
        self.convertToStandardDataset(data)

        # Compute bias
        self.computeBias()

    def convertToStandardDataset(self, data) -> None:
        # Convert Orange data to aif360 dataset, it returns a touple xdf, ydf, mdf
        xdf, ydf, mdf = data.to_pandas_dfs()
        # Merge xdf and ydf TODO: Check if I need to merge mdf
        df = ydf.merge(xdf, left_index=True, right_index=True)

        class_name = data.domain.class_var.name

        # TODO: Change this so it reads these values from the domain
        favorable_class_value = data.attributes["favorable_class_value"]
        protected_attribute = data.attributes["protected_attribute"]
        privileged_PA_values = data.attributes["privileged_PA_values"]

        # Convert the favorable_class_value and privileged_PA_values from their string representation to their integer representation
        # We need to do this because when we convert the Orange table to a pandas dataframe all categorical variables are ordinal encoded

        # Get the values for the attributes
        class_values = data.domain[class_name].values
        protected_attribute_values = data.domain[protected_attribute].values

        # Get the index of the favorable_class_value and privileged_PA_values in the list of values, this is the ordinal representation
        favorable_class_value_ordinal = class_values.index(favorable_class_value)
        privileged_PA_values_ordinal = [protected_attribute_values.index(value) for value in privileged_PA_values]
        unprivileged_PA_values_ordinal = [i for i in range(len(protected_attribute_values)) if i not in privileged_PA_values_ordinal]

        # Create the StandardDataset, this is the dataset that aif360 uses
        # df: a pandas dataframe containing all the data
        # label_name: the name of the class variable
        # favorable_classes: the values of the class variable that are considered favorable
        # protected_attribute_names: the name of the protected attribute
        # privileged_classes: the values of the protected attribute that are considered privileged (in this case they are ordinal encoded)
        self.standardDataset = StandardDataset(
            df = df,
            label_name = class_name,
            favorable_classes = [favorable_class_value_ordinal],
            protected_attribute_names = [protected_attribute],
            privileged_classes = [privileged_PA_values_ordinal],
            # categorical_features = discrete_variables,
        )

        if "weights" in mdf:
            self.standardDataset.instance_weights = mdf["weights"].to_numpy()


        # Create the privileged and unprivileged groups
        # The format is a list of dictionaries, each dictionary contains the name of the protected attribute and the ordinal value of the privileged/unprivileged group
        self.privileged_groups = [{protected_attribute: ordinal_value} for ordinal_value in privileged_PA_values_ordinal]
        self.unprivileged_groups = [{protected_attribute: ordinal_value} for ordinal_value in unprivileged_PA_values_ordinal]

    # Compute the bias of the dataset (disparate impact and statistical parity difference)
    def computeBias(self) -> None:
        dataset_metric = BinaryLabelDatasetMetric(self.standardDataset, self.unprivileged_groups, self.privileged_groups)
        self.disparate_impact = dataset_metric.disparate_impact()
        self.statistical_parity_difference = dataset_metric.statistical_parity_difference()
        self.DI_label.setText(f"Disparate Impact (ideal = 1): {self.disparate_impact}")
        self.SPD_label.setText(f"Statistical Parity Difference (ideal = 0): {self.statistical_parity_difference}")


def main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))
    args = app.arguments()
    if len(args) > 1:
        filename = args[1]
    else:
        filename = "iris"

    ow = OWDatasetBias()
    ow.show()
    ow.raise_()

    dataset = Table(filename)
    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    ow.set_data(None)
    ow.handleNewSignals()
    return 0

if __name__ == "__main__":
    sys.exit(main())