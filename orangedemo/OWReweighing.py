from typing import Optional

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.preprocess import preprocess

from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing as ReweighingAlgorithm

import pandas as pd

import sys


class Reweighing(preprocess.Preprocess):

    def __init__(self):
        self.privileged_groups = None
        self.unprivileged_groups = None
        self.standardDataset = None
        self.data = None

    def __call__(self, data):
        self.data = data
        # Create the variable "weights" using compute_value, this way the "recepie" for how to compute the weights is stored in the variable
        weights = ContinuousVariable("weights", compute_value=self.computeWeights)
        # Add the variable "weights" to the domain of the data
        new_data = data.transform(Domain(data.domain.attributes, data.domain.class_vars, data.domain.metas + (weights,)))
        return new_data

    def computeWeights(self, data):
        self.data = data
        self.TableToStandardDataset(data)
        reweighing = ReweighingAlgorithm(self.unprivileged_groups, self.privileged_groups)
        self.standardDataset = reweighing.fit_transform(self.standardDataset)
        return self.standardDataset.instance_weights

    
    # def StandardDatasetToTable(self, data, domain):
    #     # Convert aif360 StandardDataset to Orange data
    #     df = data.convert_to_dataframe()[0]
    #     # Create the dataframe of features without the class variable
    #     xdf = df.drop(columns=[self.data.domain.class_var.name])
    #     # Create the dataframe of the class variable
    #     ydf = df[[self.data.domain.class_var.name]]
    #     # Create the dataframe of the instance weights
    #     mdf = pd.DataFrame(data.instance_weights, columns=["weights"])
    #     mdf.index = xdf.index
    #     # Create the Orange table
    #     new_data = Table.from_pandas_dfs(xdf, ydf, mdf)
    #     # Set the domain of the Orange table back to the original domain
    #     # By doing so the Table is coverted from ordinal encoding back to categorical encoding
    #     # It also sets all other attributes of the domain back to the original values and adds the additional attributes back to the data
    #     weights_meta = ContinuousVariable("weights")
    #     new_metas = domain.metas + (weights_meta,)
    #     new_domain = Domain(domain.attributes, domain.class_vars, new_metas)
    #     new_data.domain = new_domain
    #     new_data.attributes = self.data.attributes
    #     # new_data.domain = domain
    #     return new_data


    def TableToStandardDataset(self, data) -> None:
        # Convert Orange data to aif360 dataset, it returns a touple xdf, ydf, mdf
        xdf, ydf, mdf = data.to_pandas_dfs()
        # Merge xdf and ydf TODO: Check if I need to merge mdf
        df = ydf.merge(xdf, left_index=True, right_index=True)

        # TODO: Change this so it reads these values from the domain
        favorable_class_value = data.attributes["favorable_class_value"]
        protected_attribute = data.attributes["protected_attribute"]
        privileged_PA_values = data.attributes["privileged_PA_values"]

        # Convert the favorable_class_value and privileged_PA_values from their string representation to their integer representation
        # We need to do this because when we convert the Orange table to a pandas dataframe all categorical variables are ordinal encoded

        # Get the values for the attributes
        class_values = data.domain.class_var.values
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
            label_name = data.domain.class_var.name,
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


class OWReweighing(OWWidget):
    name = "Reweighing"
    description = "Applies the reweighing algorithm to a dataset, which adjusts the weights of rows."
    # icon = 'icons/owreweighing.svg'
    # priority = 0

    want_main_area = False
    want_control_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Data", Table)
        

    class Outputs:
        data = Output("Preprocessed Data", Table)
        preprocessor = Output("Preprocessor", preprocess.Preprocess, dynamic=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data: Optional[Table] = None

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        if not data:
            return

        self._data = data

    def handleNewSignals(self):
        self.apply()

    def apply(self):
        if self._data is None:
            return

        reweighing = Reweighing()
        preprocessed_data = reweighing(self._data)
        self.Outputs.data.send(preprocessed_data)
        self.Outputs.preprocessor.send(reweighing)


def main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))

    # Load the data from the .tab file
    dataset = Table("./orangedemo/testDataset/adultAsFairness.tab")

    # Load the metadata from the .metadata file
    with open("./orangedemo/testDataset/adultAsFairness.tab.metadata", "r") as f:
        metadata = f.read()

    # Process the metadata and add it to the data
    # The exact way to do this will depend on the format of your metadata file
    # Here's a basic example
    # for line in metadata.split("\n"):
    #     if line:
    #         attribute_name, attribute_value = line.split("\t")
    #         attribute = dataset.domain[attribute_name]
    #         attribute.metadata = attribute_value

    ow = OWReweighing()
    ow.show()
    ow.raise_()

    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    ow.set_data(None)
    ow.handleNewSignals()
    return 0

if __name__ == "__main__":
    sys.exit(main())