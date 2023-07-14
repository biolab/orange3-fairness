from functools import wraps
from aif360.datasets import StandardDataset

from Orange.data import Domain
from Orange.widgets.utils.messages import UnboundMsg
from Orange.data import Table, Domain





MISSING_FAIRNESS_ATTRIBUTES: str = (
    'The dataset does not contain the fairness attributes. '
    'Use the "As Fairness Data" widget to add them. '
)

MISSING_ROWS: str = (
    'Rows with missing values detected. They will be omitted.'
)


def contains_fairness_attributes(domain: Domain) -> bool:
    if "favorable_class_value" not in domain.class_var.attributes:
        return False
    for var in domain:
        if "privileged_PA_values" in var.attributes:
            return True
    return False



def is_standard_dataset(data) -> bool:
    return isinstance(data, StandardDataset)


def table_to_standard_dataset(data) -> None:
    # Convert Orange data to aif360 dataset, it returns a touple xdf, ydf, mdf
    xdf, ydf, mdf = data.to_pandas_dfs()
    # Merge xdf and ydf TODO: Check if I need to merge mdf
    # This dataframe consists of all the data, the categorical variables are ordinal encoded
    df = ydf.merge(xdf, left_index=True, right_index=True)

    # Read the fairness attributes from the domain of the data
    favorable_class_value = data.domain.class_var.attributes["favorable_class_value"]
    protected_attribute = ""
    privileged_PA_values = ""
    for attribute in data.domain.attributes:
        if "privileged_PA_values" in attribute.attributes:
            protected_attribute = attribute.name
            privileged_PA_values = attribute.attributes["privileged_PA_values"]
            break

    # Convert the favorable_class_value and privileged_PA_values from their string representation to their integer representation
    # We need to do this because when we convert the Orange table to a pandas dataframe all categorical variables are ordinal encoded

    # Get the values for the attributes
    class_values = data.domain.class_var.values
    protected_attribute_values = data.domain[protected_attribute].values

    # Get the index of the favorable_class_value and privileged_PA_values in the list of values, this is the ordinal representation
    favorable_class_value_ordinal = class_values.index(favorable_class_value)
    privileged_PA_values_ordinal = [protected_attribute_values.index(value) for value in privileged_PA_values]
    unprivileged_PA_values_ordinal = [i for i in range(len(protected_attribute_values)) if i not in privileged_PA_values_ordinal]


    # If the data is from a "predict" function call and does not contain the class variable we need to add it and assign it to one of the values
    # This is because the aif360 StandardDataset requires the class variable to be present even if we will not use it so we can assign it to any value
    if data.domain.class_var.name not in df.columns:
        df[data.domain.class_var.name] = 0

    # Create the StandardDataset, this is the dataset that aif360 uses
    # df: a pandas dataframe containing all the data
    # label_name: the name of the class variable
    # favorable_classes: the values of the class variable that are considered favorable
    # protected_attribute_names: the name of the protected attribute
    # privileged_classes: the values of the protected attribute that are considered privileged (in this case they are ordinal encoded)
    standard_dataset = StandardDataset(
        df=df,
        label_name=data.domain.class_var.name,
        favorable_classes=[favorable_class_value_ordinal],
        protected_attribute_names=[protected_attribute],
        privileged_classes=[privileged_PA_values_ordinal],
        # categorical_features = discrete_variables,
    )

    if "weights" in mdf:
        standard_dataset.instance_weights = mdf["weights"].to_numpy()

    # Create the privileged and unprivileged groups
    # The format is a list of dictionaries, each dictionary contains the name of the protected attribute and the ordinal value of the privileged/unprivileged group
    privileged_groups = [{protected_attribute: ordinal_value} for ordinal_value in privileged_PA_values_ordinal]
    unprivileged_groups = [{protected_attribute: ordinal_value} for ordinal_value in unprivileged_PA_values_ordinal]

    return standard_dataset, privileged_groups, unprivileged_groups


def check_fairness_data(f):
    """Check for fairness data."""
    @wraps(f)
    def wrapper(widget, data: Table, *args, **kwargs):
        widget.Error.add_message(
            'missing_fairness_data', UnboundMsg(MISSING_FAIRNESS_ATTRIBUTES)
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
        widget.Warning.add_message('missing_values_detected', UnboundMsg(MISSING_ROWS))
        widget.Warning.missing_values_detected.clear()

        if data is not None and isinstance(data, Table):
            if data.has_missing():
                widget.Warning.missing_values_detected()

        return f(widget, data, *args, **kwargs)

    return wrapper
