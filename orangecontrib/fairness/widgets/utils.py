"""
This module contains utility functions and decorators used by the fairness widgets.
"""

import importlib.util

from functools import wraps
from aif360.datasets import StandardDataset

from Orange.widgets.utils.messages import UnboundMsg
from Orange.data import Table, Domain
from Orange.preprocess.preprocess import PreprocessorList
from Orange.preprocess import Impute

MISSING_FAIRNESS_ATTRIBUTES: str = (
    "The dataset does not contain the fairness attributes. \n"
    'Use the "As Fairness Data" widget to add them. '
)

MISSING_VALUES: str = (
    "Missing values detected in the data. \n"
    "They will automatically be imputed with the average or most frequent value."
)

MISSING_CLASS_VARIABLE: str = (
    "The dataset does not contain a class variable. \n"
    "The fairness metrics can only be used with datasets "
    "containing a (categorical) class variable. \n"
)

NUMERICAL_CLASS_VARIABLE: str = (
    "The class variable is numerical. \n"
    "The fairness metrics can only be used with categorical class variables. \n"
    "Use the discretize widget to discretize the class variable."
)

NO_CATEGORICAL_ATTRIBUTES: str = (
    "The dataset does not contain any categorical attributes except the class variable. \n"
    "The fairness metrics can only be used with datasets containing categorical attributes. \n"
    "Use the discretize widget to discretize some of the attributes."
)

REWEIGHING_PREPROCESSOR: str = (
    "This widget is not compatible with the reweighing preprocessor. \n"
    "The custom preprocessing is therefore being ignored and "
    "the default preprocessing is being used instead."
)

REWEIGHTED_DATA: str = (
    "This widget is not compatible with data which was preprocessed by a reweighing preprocessor."
)

TENSORFLOW_NOT_INSTALLED: str = (
    "The Adversarial Debiasing widget requires TensorFlow, which is not installed.\n"
    "Click on the widget to see the installation instructions."
)


def is_tensorflow_installed():
    spec = importlib.util.find_spec("tensorflow")
    return spec is not None


def check_for_tensorflow(f):
    """A function which checks if tensorflow is installed."""

    @wraps(f)
    def wrapper(widget, input, *args, **kwargs):
        if not is_tensorflow_installed():
            input = None

        return f(widget, input, *args, **kwargs)

    return wrapper


def check_for_reweighing_preprocessor(f):
    """A function which checks if the input to a widget is a reweighing preprocessor."""
    from orangecontrib.fairness.widgets.owreweighing import ReweighingTransform

    @wraps(f)
    def wrapper(widget, input, *args, **kwargs):
        # Add the preprocessor error message
        widget.Error.add_message(
            "reweighing_preprocessor", UnboundMsg(REWEIGHING_PREPROCESSOR)
        )
        widget.Error.reweighing_preprocessor.clear()

        if isinstance(input, ReweighingTransform):
            widget.Error.reweighing_preprocessor()
            input = None

        elif isinstance(input, PreprocessorList) and any(
            isinstance(p, ReweighingTransform) for p in input.preprocessors
        ):
            widget.Error.reweighing_preprocessor()
            input = None

        elif isinstance(input, Table):
            for variable in input.domain.metas:
                if variable.name == "weights":
                    widget.Error.reweighing_preprocessor()
                    input = None
                    break

        return f(widget, input, *args, **kwargs)

    return wrapper


def check_for_reweighted_data(f):
    """A function which checks if the input data war reweighted by a reweighing preprocessor."""

    @wraps(f)
    def wrapper(widget, input, *args, **kwargs):
        # Add the preprocessor error message
        widget.Error.add_message("reweighinghted_data", UnboundMsg(REWEIGHTED_DATA))
        widget.Error.reweighinghted_data.clear()

        if isinstance(input, Table):
            for variable in input.domain.metas:
                if variable.name == "weights":
                    widget.Error.reweighinghted_data()
                    input = None
                    break

        return f(widget, input, *args, **kwargs)

    return wrapper


def contains_fairness_attributes(domain: Domain) -> bool:
    """Check if the domain contains fairness attributes."""
    if domain is None or domain.class_var is None:
        return False
    if "favorable_class_value" not in domain.class_var.attributes:
        return False
    for var in domain.attributes:
        if "privileged_pa_values" in var.attributes:
            return True
    return False


def is_standard_dataset(data) -> bool:
    """Check if the data is of type standard dataset."""
    return isinstance(data, StandardDataset)


def check_fairness_data(f):
    """Wrapper function which checks if the data contains fairness attributes."""

    @wraps(f)
    def wrapper(widget, data: Table, *args, **kwargs):
        widget.Error.add_message(
            "missing_fairness_data", UnboundMsg(MISSING_FAIRNESS_ATTRIBUTES)
        )
        widget.Error.missing_fairness_data.clear()

        if data is not None and isinstance(data, Table):
            if not contains_fairness_attributes(data.domain):
                widget.Error.missing_fairness_data()
                data = None

        return f(widget, data, *args, **kwargs)

    return wrapper


def check_for_missing_values(f):
    """A wrapper function which checks if the data contains missing rows."""

    @wraps(f)
    def wrapper(widget, data: Table, *args, **kwargs):
        widget.Warning.add_message(
            "missing_values_detected", UnboundMsg(MISSING_VALUES)
        )
        widget.Warning.missing_values_detected.clear()

        if data is not None and isinstance(data, Table):
            if data.has_missing():
                widget.Warning.missing_values_detected()

        return f(widget, data, *args, **kwargs)

    return wrapper


def check_data_structure(f):
    """
    Check if the data has a (categorical) class variable and more than one categorical attribute.
    """

    @wraps(f)
    def wrapper(widget, data: Table, *args, **kwargs):
        widget.Error.add_message(
            "missing_class_variable", UnboundMsg(MISSING_CLASS_VARIABLE)
        )
        widget.Error.add_message(
            "numerical_class_variable", UnboundMsg(NUMERICAL_CLASS_VARIABLE)
        )
        widget.Error.add_message(
            "no_categorical_attributes", UnboundMsg(NO_CATEGORICAL_ATTRIBUTES)
        )

        widget.Error.missing_class_variable.clear()
        widget.Error.numerical_class_variable.clear()
        widget.Error.no_categorical_attributes.clear()

        if data is not None and isinstance(data, Table):
            if not data.domain.class_var:
                widget.Error.missing_class_variable()
                data = None

            elif not data.domain.class_var.is_discrete:
                widget.Error.numerical_class_variable()
                data = None

            if data is not None:
                categorical = False
                for attribute in data.domain.attributes:
                    if attribute.is_discrete:
                        categorical = True
                        break

                if not categorical:
                    widget.Error.no_categorical_attributes()
                    data = None

        return f(widget, data, *args, **kwargs)

    return wrapper


#############################################################
# Functions related to the table_to_standard_dataset function
#############################################################


def _get_fairness_attributes(data):
    """Get the fairness attributes from the domain of the data."""
    favorable_class_value = data.domain.class_var.attributes["favorable_class_value"]
    protected_attribute = ""
    privileged_pa_values = ""
    for attribute in data.domain.attributes:
        if "privileged_pa_values" in attribute.attributes:
            protected_attribute = attribute.name
            privileged_pa_values = attribute.attributes["privileged_pa_values"]
            break
    return favorable_class_value, protected_attribute, privileged_pa_values


def _get_index_attribute_encoding(
    data, protected_attribute, favorable_class_value, privileged_pa_values
):
    """
    Convert the favorable_class_value and privileged_pa_values
    from their string representation to their index representation
    """
    # Get the values for the attributes
    class_values = data.domain.class_var.values
    protected_attribute_values = data.domain[protected_attribute].values

    # Get the index representation of the favorable_class_value and privileged_pa_values,
    # this is their index in the list of values.
    favorable_class_value_indexes = class_values.index(favorable_class_value)
    privileged_pa_values_indexes = [
        protected_attribute_values.index(value) for value in privileged_pa_values
    ]
    unprivileged_pa_values_indexes = [
        i
        for i in range(len(protected_attribute_values))
        if i not in privileged_pa_values_indexes
    ]
    return (
        favorable_class_value_indexes,
        privileged_pa_values_indexes,
        unprivileged_pa_values_indexes,
    )


def _add_dummy_class_column(data, df):
    """Create an array with index class variable values with the same length as the dataframe"""
    num_unique_values = len(data.domain.class_var.values)
    repeated_values = list(range(num_unique_values)) * (
        len(df) // num_unique_values + 1
    )
    df[data.domain.class_var.name] = repeated_values[: len(df)]


def _correct_standard_dataset(standard_dataset, favorable_class_value_indexes):
    """
    Check if the favorable_label in the standard_dataset matches the
    favorable_class_value_indexes (the expected favorable label value) and correct it if needed.
    """
    if standard_dataset.favorable_label != favorable_class_value_indexes:
        standard_dataset.favorable_label, standard_dataset.unfavorable_label = (
            standard_dataset.unfavorable_label,
            standard_dataset.favorable_label,
        )


def table_to_standard_dataset(data) -> None:
    """Converts an Orange.data.Table to an aif360 StandardDataset."""

    if not contains_fairness_attributes(data.domain):
        raise ValueError(MISSING_FAIRNESS_ATTRIBUTES)

    if data.has_missing():
        data = Impute()(data)

    xdf, ydf, mdf = data.to_pandas_dfs()
    # Merge xdf and ydf (currently not merging mdf because it is not used)
    # This dataframe consists of all the data, the categorical variables values are
    # represented with the index of the value in domain[attribute].values
    df = ydf.merge(xdf, left_index=True, right_index=True)

    # Read the fairness attributes from the domain of the data,
    # which will be used to get the index representations
    (
        favorable_class_value,
        protected_attribute,
        privileged_pa_values,
    ) = _get_fairness_attributes(data)

    # Convert the favorable_class_value and privileged_pa_values from their string
    # representation to their index representation. We need to do this because when
    # we convert the Orange table to a pandas dataframe all categorical variables are encoded.
    (
        favorable_class_value_indexes,
        privileged_pa_values_indexes,
        _,
    ) = _get_index_attribute_encoding(
        data, protected_attribute, favorable_class_value, privileged_pa_values
    )

    # If the data is from a "predict" function call and does not contain the class
    # variable we need to add it and fill it with dummy values. The dummy values need
    # to contain all the possible values of the class variable (in its index representation).
    # This is because the aif360 StandardDataset requires the class variable to be present in
    # the dataframe with all the possible values.
    if (
        data.domain.class_var.name not in df.columns
        or df[data.domain.class_var.name].isnull().any()
    ):
        _add_dummy_class_column(data, df)

    # Map the protected_attribute privileged values to 1 and the unprivileged values to 0
    # This is so AdversarialDebiasing can work when the protected attribute has more than
    # two unique values. It does not affect the performance of any other algorithm.
    df[protected_attribute] = df[protected_attribute].map(
        lambda x: 1 if x in privileged_pa_values_indexes else 0
    )

    # Create the StandardDataset, this is the dataset that aif360 uses
    standard_dataset = StandardDataset(
        df=df,  # df: a pandas dataframe containing all the data
        label_name=data.domain.class_var.name,  # label_name: the name of the class variable
        favorable_classes=[
            favorable_class_value_indexes
        ],  # favorable_classes: the values of the class variable that are considered favorable
        protected_attribute_names=[
            protected_attribute
        ],  # protected_attribute_names: the name of the protected attribute
        privileged_classes=[
            [1]
        ],  # privileged_classes: the values of the protected attribute that are considered
        # privileged (in this case they are index encoded).
        # categorical_features = discrete_variables,
    )

    # Adversarial debiasing bug fix (in the prediction phase when using Average Impute
    # and Predictions widget all labels are set to the same value for some reason.
    # This messes with the standard dataset because it thinks the label is not binary
    # and binarizes it on its own, and selects its own favorable_label which is not what
    # we want or expect so in case that happens we want to set the favorable_label back to
    # what it should be) (This could also apply to some other scenarios so it is better to
    # have it here as a precaution)
    _correct_standard_dataset(standard_dataset, favorable_class_value_indexes)

    if "weights" in mdf:
        standard_dataset.instance_weights = mdf["weights"].to_numpy()

    # Create the privileged and unprivileged groups
    # The format was a list of dictionaries, each dictionary contains the name of the protected
    # attribute and the index value of the privileged/unprivileged group.
    # AdversaryDebiasing can only handle one protected attribute, so we converted all privileged
    # values to 1 and unprivileged to 0 and now only need one dictionary (the result is the same).
    privileged_groups = [{protected_attribute: 1}]
    unprivileged_groups = [{protected_attribute: 0}]

    return standard_dataset, privileged_groups, unprivileged_groups
