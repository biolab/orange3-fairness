"""
This module contains the implementation of the Reweighing widget.

This widget applies the reweighing algorithm to a dataset, which adjusts the weights of rows.
"""

from typing import Optional

from Orange.widgets import gui
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data import Table, Domain, ContinuousVariable
from Orange.preprocess import preprocess

from aif360.algorithms.preprocessing import Reweighing as ReweighingAlgorithm

from orangecontrib.fairness.widgets.utils import (
    table_to_standard_dataset,
    check_fairness_data,
    check_for_missing_values,
)


class MzCom:
    """
    A class used to compute the weights of the rows of a
    dataset using a already fitted reweighing algorithm
    """

    def __init__(self, model, original_domain=None):
        self.original_domain = original_domain
        self.model = model

    def __call__(self, data):
        # For creating the standard dataset we need to know the encoding the table uses for
        # the class variable. This can be found in the domain and is the same as the order
        # of values of the class variable in the domain. This is why we need to add it back
        # to the domain if it was removed.
        if not data.domain.class_var:
            data.domain.class_var = self.original_domain.class_var
        data, _, _ = table_to_standard_dataset(data)
        data = self.model.transform(data)
        return data.instance_weights

    InheritEq = True


class ReweighingModel:
    """
    A class used to create a ReweighingAlgoritm instance, fitting it to the data and returning it.
    """

    def __call__(self, data):
        (
            standard_dataset,
            privileged_groups,
            unprivileged_groups,
        ) = table_to_standard_dataset(data)
        reweighing = ReweighingAlgorithm(unprivileged_groups, privileged_groups)
        reweighing = reweighing.fit(standard_dataset)
        return reweighing


class ReweighingTransform(preprocess.Preprocess):
    """
    A class used to add a new column/variable to the data with the weights of
    the rows of the data computed by the fitted reweighing algorithm stored in
    the MzCom class instance as a compute_value function.
    """

    def __call__(self, data):
        model = ReweighingModel()(data)
        weights = ContinuousVariable(
            "weights", compute_value=MzCom(model, original_domain=data.domain)
        )
        # Alternative for the compute_value:
        # compute_value=lambda data, model=model: transf(data, model)

        # Add the variable "weights" to the domain of the data
        new_data = data.transform(
            Domain(
                data.domain.attributes,
                data.domain.class_vars,
                data.domain.metas + (weights,),
            )
        )
        return new_data


class OWReweighing(OWWidget):
    """
    A class used to create a widget for the reweighing algorithm,
    which can be used to transform a dataset or as a preprocessor for a model.
    """

    name = "Reweighing"
    description = (
        "Applies the reweighing algorithm to a dataset, "
        "which adjusts the weights of rows."
    )
    icon = "icons/reweighing.svg"
    priority = 20

    want_control_area = False
    resizing_enabled = False

    class Inputs:
        """The inputs of the widget - the dataset"""

        data = Input("Data", Table)

    class Outputs:
        """The outputs of the widget - the preprocessed dataset and the preprocessor"""

        data = Output("Preprocessed Data", Table)
        preprocessor = Output("Preprocessor", preprocess.Preprocess, dynamic=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = ReweighingTransform()
        self.Outputs.preprocessor.send(self.preprocessor)

        box = gui.vBox(self.mainArea, "Info")
        gui.widgetLabel(
            box,
            (
                "This widget applies the reweighing algorithm to a dataset, "
                "which adjusts the weights of rows.\nThe input data must have "
                "the additional 'AsFairness' attributes and be without any missing values."
            ),
        )

        self._data: Optional[Table] = None

    @Inputs.data
    @check_fairness_data
    @check_for_missing_values
    def set_data(self, data: Optional[Table]) -> None:
        """Handling the input data by saving it"""
        if not data:
            return

        self._data = data

    def handleNewSignals(self):
        """Handling any new signals by applying the reweighing algorithm to the data"""
        self.apply()

    def apply(self):
        """
        Fitting the reweighing algorithm to the data and sending the
        preprocessed data and the preprocessor to the output.
        """
        if self._data is None:
            return

        preprocessed_data = self.preprocessor(self._data)

        self.Outputs.data.send(preprocessed_data)
        self.Outputs.preprocessor.send(self.preprocessor)
