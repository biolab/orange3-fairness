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


def table_to_standard_dataset(data) -> None:
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
        standardDataset = StandardDataset(
            df = df,
            label_name = data.domain.class_var.name,
            favorable_classes = [favorable_class_value_ordinal],
            protected_attribute_names = [protected_attribute],
            privileged_classes = [privileged_PA_values_ordinal],
            # categorical_features = discrete_variables,
        )

        if "weights" in mdf:
            standardDataset.instance_weights = mdf["weights"].to_numpy()

        # Create the privileged and unprivileged groups
        # The format is a list of dictionaries, each dictionary contains the name of the protected attribute and the ordinal value of the privileged/unprivileged group
        privileged_groups = [{protected_attribute: ordinal_value} for ordinal_value in privileged_PA_values_ordinal]
        unprivileged_groups = [{protected_attribute: ordinal_value} for ordinal_value in unprivileged_PA_values_ordinal]

        return standardDataset, privileged_groups, unprivileged_groups



class MzCom:
    # The __init__ method is called when the class is created and can have as many arguments as you want. MzCom(model) creates an instance of the class
    # The __call__ method is called when the class is called, it must only have one argument, which is the data. MzCom(model)(data) calls the __call__ method of the class
    def __init__(self, model):
        self.model = model

    def __call__(self, data):
        if not isinstance(data, StandardDataset):
            data, _, _ = table_to_standard_dataset(data)
        # Check if the model is a ReweighingAlgorithm, if not raise an error
        if not isinstance(self.model, ReweighingAlgorithm):
            raise ValueError("The model must be a ReweighingAlgorithm")
        # Call the transform method of the model
        data = self.model.transform(data)
        # Return the weights
        return data.instance_weights
    
class ReweighingModel(preprocess.Preprocess):
    # This class doesn't need an __init__ method because it doesn't need any arguments when it is created
    # The __call__ method creates an instance of the ReweighingAlgorithm, fits it to the data and returns it
    def __call__(self, data):
        standardDataset, privileged_groups, unprivileged_groups = table_to_standard_dataset(data)
        reweighing = ReweighingAlgorithm(unprivileged_groups, privileged_groups)
        reweighing = reweighing.fit(standardDataset)
        return reweighing
    

class ReweighingTransform(preprocess.Preprocess):
    # The __call__ method applies the reweighing algorithm to the data and returns the data with the weights
    def __call__(self, data):
        # Create an instalce of the ReweighingModel, and call the __call__ method with the data as argument
        model = ReweighingModel()(data)
        # Create a new variable "weights" with the compute_value function, the compute_value function is the MzCom class, which when called calls the transform method of the model
        weights = ContinuousVariable("weights", compute_value=MzCom(model))
        # Alternative for the compute_value: compute_value=lambda data, model=model: transf(data, model)
        
        # Add the variable "weights" to the domain of the data
        new_data = data.transform(Domain(data.domain.attributes, data.domain.class_vars, data.domain.metas + (weights,)))
        return new_data
    

    
class OWReweighing(OWWidget):
    name = "Reweighing"
    description = "Applies the reweighing algorithm to a dataset, which adjusts the weights of rows."
    # icon = 'icons/owreweighing.svg'
    # priority = 0

    want_control_area = False
    resizing_enabled = False

    # Define the inputs and outputs of the widget
    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Preprocessed Data", Table)
        preprocessor = Output("Preprocessor", preprocess.Preprocess, dynamic=False)

    # Define the initial state of the widget (constructor)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        box = gui.vBox(self.mainArea, "Info")
        gui.widgetLabel(box, "This widget applies the reweighing algorithm to a dataset, which adjusts the weights of rows.\nThe input data must have the additional 'AsFairness' attributes and be without any missing values.")

        self._data: Optional[Table] = None

    # Define what should happen when the input data is received
    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        if not data:
            return

        self._data = data

    def handleNewSignals(self):
        self.apply()

    #
    def apply(self):
        if self._data is None:
            return

        preprocessor = ReweighingTransform()
        preprocessed_data = preprocessor(self._data)

        self.Outputs.data.send(preprocessed_data)
        self.Outputs.preprocessor.send(preprocessor)
