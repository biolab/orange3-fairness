from aif360.datasets import StandardDataset


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