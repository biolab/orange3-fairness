from functools import wraps
from aif360.datasets import StandardDataset

from Orange.data import Domain
from Orange.widgets.utils.messages import UnboundMsg
from Orange.data import Table, Domain


MISSING_FAIRNESS_ATTRIBUTES: str = (
    "The dataset does not contain the fairness attributes. "
    'Use the "As Fairness Data" widget to add them. '
)

MISSING_ROWS: str = "Rows with missing values detected. They will be omitted."


def contains_fairness_attributes(domain: Domain) -> bool:
    if "favorable_class_value" not in domain.class_var.attributes:
        return False
    for var in domain:
        if "privileged_PA_values" in var.attributes:
            return True
    return False


def is_standard_dataset(data) -> bool:
    return isinstance(data, StandardDataset)


def check_fairness_data(f):
    """Check for fairness data."""

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


def check_for_missing_rows(f):
    """Check for missing rows."""

    @wraps(f)
    def wrapper(widget, data: Table, *args, **kwargs):
        widget.Warning.add_message("missing_values_detected", UnboundMsg(MISSING_ROWS))
        widget.Warning.missing_values_detected.clear()

        if data is not None and isinstance(data, Table):
            if data.has_missing():
                widget.Warning.missing_values_detected()

        return f(widget, data, *args, **kwargs)

    return wrapper


#############################################################
# Functions related to the table_to_standard_dataset function
#############################################################


def _get_fairness_attributes(data):
    # Read the fairness attributes from the domain of the data
    favorable_class_value = data.domain.class_var.attributes["favorable_class_value"]
    protected_attribute = ""
    privileged_PA_values = ""
    for attribute in data.domain.attributes:
        if "privileged_PA_values" in attribute.attributes:
            protected_attribute = attribute.name
            privileged_PA_values = attribute.attributes["privileged_PA_values"]
            break
    return favorable_class_value, protected_attribute, privileged_PA_values


def _get_index_attribute_encoding(
    data, protected_attribute, favorable_class_value, privileged_PA_values
):
    # Convert the favorable_class_value and privileged_PA_values from their string representation to their index representation
    # Get the values for the attributes
    class_values = data.domain.class_var.values
    protected_attribute_values = data.domain[protected_attribute].values

    # Get the index representation of the favorable_class_value and privileged_PA_values, this is their index in the list of values
    favorable_class_value_indexes = class_values.index(favorable_class_value)
    privileged_PA_values_indexes = [
        protected_attribute_values.index(value) for value in privileged_PA_values
    ]
    unprivileged_PA_values_indexes = [
        i
        for i in range(len(protected_attribute_values))
        if i not in privileged_PA_values_indexes
    ]
    return (
        favorable_class_value_indexes,
        privileged_PA_values_indexes,
        unprivileged_PA_values_indexes,
    )


def _add_dummy_class_column(data, df):
    # Create an array with index class variable values with the same length as the dataframe
    num_unique_values = len(data.domain.class_var.values)
    repeated_values = list(range(num_unique_values)) * (
        len(df) // num_unique_values + 1
    )
    # Add the dummy class column to the dataframe
    df[data.domain.class_var.name] = repeated_values[: len(df)]


def table_to_standard_dataset(data) -> None:
    """Converts an Orange.data.Table to an aif360 StandardDataset."""

    if not contains_fairness_attributes(data.domain):
        raise ValueError(MISSING_FAIRNESS_ATTRIBUTES)
    # Convert Orange data to aif360 dataset, it returns a touple xdf, ydf, mdf
    xdf, ydf, mdf = data.to_pandas_dfs()
    # Merge xdf and ydf TODO: Check if I need to merge mdf
    # This dataframe consists of all the data, the categorical variables values are represented with the index of the value in domain[attribute].values
    df = ydf.merge(xdf, left_index=True, right_index=True)

    # Read the fairness attributes from the domain of the data, which will be used to get the index representations
    (
        favorable_class_value,
        protected_attribute,
        privileged_PA_values,
    ) = _get_fairness_attributes(data)

    # Convert the favorable_class_value and privileged_PA_values from their string representation to their index representation
    # We need to do this because when we convert the Orange table to a pandas dataframe all categorical variables are encoded
    (
        favorable_class_value_indexes,
        privileged_PA_values_indexes,
        unprivileged_PA_values_indexes,
    ) = _get_index_attribute_encoding(
        data, protected_attribute, favorable_class_value, privileged_PA_values
    )

    # If the data is from a "predict" function call and does not contain the class variable we need to add it and fill it with dummy values
    # The dummy values need to contain all the possible values of the class variable (in its index representation)
    # This is because the aif360 StandardDataset requires the class variable to be present in the dataframe with all the possible values
    if data.domain.class_var.name not in df.columns:
        _add_dummy_class_column(data, df)


    # Map the protected_attribute privileged values to 1 and the unprivileged values to 0
    # This is so AdversarialDebiasing can work when the protected attribute has more than two unique values
    # It does not affect the performance of any other algorithm
    df[protected_attribute] = df[protected_attribute].map(
        lambda x: 1 if x in privileged_PA_values_indexes else 0
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
        ],  # privileged_classes: the values of the protected attribute that are considered privileged (in this case they are index encoded)
        # categorical_features = discrete_variables,
    )

    if "weights" in mdf:
        standard_dataset.instance_weights = mdf["weights"].to_numpy()

    # Create the privileged and unprivileged groups
    # The format was a list of dictionaries, each dictionary contains the name of the protected attribute and the index value of the privileged/unprivileged group
    # Because AdversaryDebiasing can only handle one protected attribute, we converted all privileged values to 1 and unprivileged to 0 and now only need one dictionary (the result is the same)
    privileged_groups = [
        {protected_attribute: 1}
    ]
    unprivileged_groups = [
        {protected_attribute: 0}
    ]


    return standard_dataset, privileged_groups, unprivileged_groups