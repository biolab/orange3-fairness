from typing import Optional

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel
from Orange.data import Table, Domain, DiscreteVariable

from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.fairness.widgets.utils import check_data_structure


class OWAsFairness(OWWidget):
    """
    Converts a dataset to a fairness dataset with marked favorable class values, 
    protected attributes and priviliged protected attribute values.
    """
    name = "As Fairness Data"
    description = "Converts a dataset to a fairness dataset with marked favorable class values, protected attributes and priviliged protected attribute values."
    icon = 'icons/as_fairness.svg'
    priority = 0

    want_main_area = False
    resizing_enabled = False

    class Inputs:
        """Define the inputs to the widgets"""
        data = Input("Data", Table)

    class Outputs:
        """Define the outputs to the widgets"""
        data = Output("Data", Table)

    # Settings: The favorable_class, protected_attribute, and privileged_values are instance variables that are declared as ContextSetting.
    # A ContextSetting is a special type of Setting that Orange remembers for each different context (i.e., input data domain).
    settingsHandler = DomainContextHandler(
        match_values=DomainContextHandler.MATCH_VALUES_ALL
    )
    protected_attribute = ContextSetting(None)
    favorable_class_value = ContextSetting("")
    privileged_pa_values = ContextSetting([])
    auto_commit: bool = Setting(
        True, schema_only=True
    )  # schema_only -> The setting is saved within the workflow but the default never changes.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data: Optional[Table] = None
        favorable_class_items_model = PyListModel(
            iterable=[]
        )  # Here we don't want to display the different attributes/features but the values of the attribute so I couldn't use the DomainModel which stores the features, so I used the PyListModel which can store any type in an iterable
        protected_attribute_model = DomainModel(
            valid_types=(DiscreteVariable,)
        )  # DomainModel stores the domain of the input data and allows us to select a attribute from it. It is tied to the gui.comboBox widget. And Can be accesed by self.controls.protected_attribute.model()
        privileged_pa_values_model = PyListModel(iterable=[])

        # Create a box for each of the three variables and populate them with comboboxes and listboxes.
        box = gui.vBox(self.controlArea, "Favorable Class Value", margin=0)
        gui.comboBox(
            box,
            self,
            "favorable_class_value",
            model=favorable_class_items_model,
            callback=self.commit.deferred,
            searchable=True,
        )

        box = gui.vBox(self.controlArea, "Protected Attribute", margin=0)
        gui.comboBox(
            box,
            self,
            "protected_attribute",
            model=protected_attribute_model,
            callback=[self.change_values, self.commit.deferred],
            searchable=True,
        )

        box = gui.vBox(
            self.controlArea, "Privileged Protected Attribute Values", margin=0
        )
        var_list = gui.listView(
            box,
            self,
            "privileged_pa_values",
            model=privileged_pa_values_model,
            callback=self.commit.deferred,
        )
        var_list.setSelectionMode(var_list.ExtendedSelection)

        # A Commit/AutoCommit button which controls if a new signal is sent whenever the user changes the value of a variable or only when the user commits the changes
        self.commit_button = gui.auto_commit(
            self.controlArea, self, "auto_commit", "&Commit", box=False
        )

    @Inputs.data
    @check_data_structure
    def set_data(self, data: Table) -> None:
        """
        Function called when new data is received on the input. It is responsible for filling 
        the widget comboboxes and listboxes with the values contained in the input data.
        """
        self.closeContext()
        self._data = None
        domain: Optional[Domain] = None

        if data is not None:
            # Remove all rows with missing values from the data, because the fairness algorithms can't handle missing values.

            # Create a table with the same data but a different domain
            self._data = data.transform(data.domain)
            # Copy the attributes from the original data to the new data
            self._data.attributes = data.attributes.copy()

            # A new Domain object is created from the attributes and class variables of the input data.
            # This Domain object represents the structure of the input data (i.e., the variables it contains).
            domain = Domain(data.domain.attributes + data.domain.class_vars)

            # Changes the domain of the comboboxes to the new domain, thus updating the list of variables that can be selected.
            self.controls.protected_attribute.model().set_domain(domain)

            # Clear the old values of the favorable_class_items_model PyListModel
            if self.controls.favorable_class_value.model():
                self.controls.favorable_class_value.model().clear()
            # Fill the favorable_class_items_model PyListModel with the values of the class variable of the input data.
            for value in data.domain.class_vars[0].values:
                self.controls.favorable_class_value.model().append(value)

            # Clear the old values of the privileged_pa_values_model PyListModel
            if self.controls.privileged_pa_values.model():
                self.controls.privileged_pa_values.model().clear()
            # Fill the privileged_pa_values_model PyListModel with the values of the protected attribute variable of the input data.
            for value in self.controls.protected_attribute.model()[0].values:
                self.controls.privileged_pa_values.model().append(value)

            # Set the selected variables to the first variable in the list of variables (in the comboboxes or listboxes)
            self.protected_attribute = (
                self.controls.protected_attribute.model()[0]
                if len(self.controls.protected_attribute.model())
                else None
            )
            self.favorable_class_value = (
                self.controls.favorable_class_value.model()[0]
                if len(self.controls.favorable_class_value.model())
                else None
            )
            self.privileged_pa_values = (
                [self.controls.privileged_pa_values.model()[0]]
                if len(self.controls.privileged_pa_values.model())
                else []
            )

            # If atleast one value for each variable is selected, then open the context for the widget using the new domain
            # This means that these settings will be remembered the next time the widget receives the same input data.
            # If the input data allready has a known domain, then the saved settings will be used.
            if (
                self.protected_attribute is not None
                and self.favorable_class_value is not None
                and len(self.controls.privileged_pa_values.model()) > 0
            ):
                self.openContext(domain)

        # Apply the changes and send the data to the output
        # Here we have commit.now() instead of a commit.deferred() because we want to apply the changes as soon as the input data changes.
        self.commit.now()

    def openContext(self, *a):
        """
        Call the openContext function of the parent class and then check if the protected
        attribute variable or the privileged_pa_values_model PyListModel need to be changed.
        """
        super().openContext(*a)

        # Check if the privileged_pa_values match the values of the protected attribute variable
        # This is needed because when loading a old workflow, the domain sometimes doesn't change the protected_attribute
        if not set(self.privileged_pa_values).issubset(set(self.protected_attribute.values)):
            self.change_values(clear_pa_values=True)
        # Check if the self.controls.privileged_pa_values.model matches the values of the protected attribute variable
        # This is needed when loading an old workflow the displayed values might not match the values of the protected attribute variable
        elif not set(self.controls.privileged_pa_values.model()).issubset(
            set(self.protected_attribute.values)
        ):
            self.change_values(clear_pa_values=False)


    def change_values(self, clear_pa_values=True) -> None:
        """
        This function is normally called when the user changes the protected attribute variable
        It changes the values of the privileged_pa_values_model PyListModel (the list of displayed privileged PA values) 
        and the selected privileged PA values (self.privileged_pa_values) to match the values of the new protected attribute variable.
        """

        # Change the list of displayed privileged PA values
        self.controls.privileged_pa_values.model().clear()
        # Fill the list with the values of the new protected attribute variable
        for value in self.protected_attribute.values:
            self.controls.privileged_pa_values.model().append(value)

        # Change the selected privileged PA values
        if clear_pa_values:
            self.privileged_pa_values = (
                [self.controls.privileged_pa_values.model()[0]]
                if len(self.controls.privileged_pa_values.model())
                else []
            )


    # Adding the protected attribute and favorable class value as attributes to the data domain
    def as_fairness_data(self, data: Table) -> Optional[Table]:
        """This function adds the protected attribute and favorable class value as attributes to the data domain"""
        if (
            not self.protected_attribute
            or not self.favorable_class_value
            or not self.privileged_pa_values
            or not data
        ):
            return None

        old_domain = data.domain

        # Create the new attribute or copy the old ones
        new_attributes = []
        for attribute in old_domain.attributes:
            # If the attribute is the protected attribute, then create a new attribute with the same values but add the privileged_pa_values attribute
            if attribute.name == self.protected_attribute.name:
                new_attr = attribute.copy()
                new_attr.attributes["privileged_pa_values"] = self.privileged_pa_values
            # Else just copy the attribute
            else:
                new_attr = attribute
            new_attributes.append(new_attr)

        # Create new class_var
        new_class_var = old_domain.class_var.copy()
        new_class_var.attributes["favorable_class_value"] = self.favorable_class_value

        new_domain = Domain(new_attributes, new_class_var, old_domain.metas)

        # Create a new table with the new domain
        new_data = data.transform(new_domain)
        return new_data

    @gui.deferred  # The defered allows us to only call the function once the user has stopped changing the values of the comboboxes or listboxes and "Applies" the changes
    def commit(self) -> None:
        """
        This function is called when the user changes the value of the comboboxes or listboxes.
        It changes the data attributes and outputs the data
        """
        data = None
        if self._data is not None:
            data = self.as_fairness_data(self._data)
        self.Outputs.data.send(data)


if __name__ == "__main__":
    table = Table("zoo.tab")
    WidgetPreview(OWAsFairness).run(input_data=table)
