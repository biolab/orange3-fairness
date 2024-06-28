"""
This module contains the AsFairness widget which is used to add fairness attributes to the data.

The fairness attributes are the favorable class value, 
protected attribute and privileged protected attribute values.
"""

from typing import Optional

from AnyQt.QtCore import QItemSelectionModel

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel
from Orange.data import Table, Domain, DiscreteVariable

from orangecontrib.fairness.widgets.utils import check_data_structure


class OWAsFairness(OWWidget):
    """
    Converts a dataset to a fairness dataset with marked favorable class values,
    protected attributes and priviliged protected attribute values.
    """

    name = "As Fairness Data"
    description = (
        "Converts a dataset to a fairness dataset with marked favorable class values, "
        "protected attributes and priviliged protected attribute values."
    )
    icon = "icons/as_fairness_data.svg"
    priority = 0

    want_main_area = False
    resizing_enabled = True

    class Inputs:
        """Define the inputs to the widgets"""

        data = Input("Data", Table)

    class Outputs:
        """Define the outputs to the widgets"""

        data = Output("Data", Table)

    # Settings: The favorable_class, protected_attribute, and privileged_values
    # are instance variables that are declared as ContextSetting.
    # A ContextSetting is a special type of Setting that Orange remembers for
    # each different context (i.e., input data domain).
    settingsHandler = DomainContextHandler(
        match_values=DomainContextHandler.MATCH_VALUES_ALL
    )
    protected_attribute = ContextSetting(None)
    favorable_class_value = ContextSetting("")
    privileged_pa_values = ContextSetting([])
    auto_commit: bool = Setting(
        True, schema_only=True
    )  # schema_only -> The setting is saved within the workflow but the default never changes.
    keep_default: bool = Setting(
        True, schema_only=True
    )  # This setting will be used to keep the original fairness attributes when first using the
    # widget, If the user changes the attributes from the default ones, we want to use the context

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data: Optional[Table] = None
        self.original_favorable_class_value = None
        self.original_protected_attribute = None
        self.original_privileged_pa_values = None

        favorable_class_items_model = PyListModel(
            iterable=[]
        )  # Here we don't want to display the different attributes/features but the values of the
        # attribute so I couldn't use the DomainModel which stores the features,
        # so I used the PyListModel which can store any type in an iterable.
        protected_attribute_model = DomainModel(
            valid_types=(DiscreteVariable,)
        )  # DomainModel stores the domain of the input data and allows us to select a attribute
        # from it. It is tied to the gui.comboBox widget. And Can be accesed by
        # self.controls.protected_attribute.model()
        privileged_pa_values_model = PyListModel(iterable=[])

        # Create a box for each of the three variables, populate them with comboboxes and listboxes.
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

        # A Commit/AutoCommit button which controls if a new signal is sent whenever the user
        # changes the value of a variable or only when the user commits the changes
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
            # Create a table with the same data but a different domain
            self._data = data.transform(data.domain)
            # Copy the attributes from the original data to the new data
            self._data.attributes = data.attributes.copy()

            # A new Domain object is created from the attributes and class variables of the
            # input data. This Domain object represents the structure of the input data
            # (i.e., the variables it contains).
            domain = Domain(data.domain.attributes, data.domain.class_vars)

            # Find the original fairness attributes in the input data and save them.
            self.find_and_read_fainress_attributes(domain)

            # Clear the comboboxes and listboxes and fill them with the values of the input data.
            self._clear_and_fill(domain)

            # Open the context for the widget using the new domain
            # This means that these settings will be remembered the next time the widget receives
            # the same input data. If the input data allready has a known domain, then the saved
            # settings will be used.
            self.openContext(domain)

        # Apply the changes and send the data to the output
        # Here we have commit.now() instead of a commit.deferred() because
        # we want to apply the changes as soon as the input data changes.
        self.commit.now()

    def _clear_and_fill(self, domain):
        """
        This function is used to first clear the comboboxes and listboxes of their values
        and then fill them with the values of the input data.
        """

        # Changes the domain of the comboboxes to the new domain,
        # thus updating the list of variables that can be selected.
        self.controls.protected_attribute.model().set_domain(Domain(domain.attributes))

        # Clear the old values of the favorable_class_items_model PyListModel
        if self.controls.favorable_class_value.model():
            self.controls.favorable_class_value.model().clear()
        # Fill the favorable_class_items_model PyListModel with the
        # values of the class variable of the input data.
        for value in domain.class_vars[0].values:
            self.controls.favorable_class_value.model().append(value)

        # Clear the old values of the privileged_pa_values_model PyListModel
        if self.controls.privileged_pa_values.model():
            self.controls.privileged_pa_values.model().clear()
        # Fill the privileged_pa_values_model PyListModel with the values
        # of the protected attribute variable of the input data.
        for value in self.controls.protected_attribute.model()[0].values:
            self.controls.privileged_pa_values.model().append(value)

        # Select the first values of the comboboxes and listboxes.
        self._select_values()

    def _select_values(self):
        """
        This function is used to set the values of the comboboxes and listboxes to the first value
        of the list of variables. This is needed so that the user doesn't have to select every value
        manually even if it is the first value of the list and it is already shown as the selected
        value in the dropdown menu causing confusion.
        """

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

    def find_and_read_fainress_attributes(self, domain):
        """
        This function is used to find the original fairness attributes in input data and save them
        as instance variables. This is needed because the openContext will overwrite the fairness
        attributes when it is called and sometimes we want to keep the original ones.
        """

        # Check if the data contains fairness attributes if so then set
        # the selected variables to the values of the fairness attributes
        if "favorable_class_value" in domain.class_var.attributes:
            # Set the selected favorable class value to the value of the key "favorable_class_value"
            self.original_favorable_class_value = domain.class_var.attributes[
                "favorable_class_value"
            ]

            for var in domain.attributes:
                if "privileged_pa_values" in var.attributes:
                    self.original_protected_attribute = var
                    self.original_privileged_pa_values = var.attributes[
                        "privileged_pa_values"
                    ]
                    break

    def openContext(self, *a):
        """
        Call the openContext function of the parent class and then check if the protected
        attribute variable or the privileged_pa_values_model PyListModel need to be changed.
        """
        super().openContext(*a)

        # If the user is loading the dataset for the first time and the dataset has some
        # default fairness attributes, then we want to keep the default fairness attributes
        # and not overwrite them with openContext.
        if (
            self.keep_default
            and self.original_favorable_class_value
            and self.original_protected_attribute
            and self.original_privileged_pa_values
        ):
            self.favorable_class_value = self.original_favorable_class_value
            self.protected_attribute = self.original_protected_attribute
            self.privileged_pa_values = self.original_privileged_pa_values

        # Check if the privileged_pa_values match the values of the protected attribute variable
        # This is needed because when loading a old workflow, the openContext sometimes doesn't
        # change the protected_attribute
        if not set(self.privileged_pa_values).issubset(
            set(self.protected_attribute.values)
        ):
            self.change_values(clear_pa_values=True)

        # Check if the values in self.controls.privileged_pa_values.model matches the values of
        # the protected attribute variable. This is needed when loading an old workflow the
        # displayed values might not match the values of the protected attribute variable.
        elif not set(self.controls.privileged_pa_values.model()).issubset(
            set(self.protected_attribute.values)
        ):
            self.change_values(clear_pa_values=False)

    def change_values(self, clear_pa_values=True) -> None:
        """
        This function is normally called when the user changes the protected attribute variable
        It changes the values of the privileged_pa_values_model PyListModel (the list of displayed
        privileged PA values) and the selected privileged PA values (self.privileged_pa_values)
        to match the values of the new protected attribute variable.
        """

        # Context is now needed to use the selected fairness attributes instead
        # of the default ones if the user saves and loads the workflow
        self.keep_default = False

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
        # If we don't want to clear the selected privileged PA values,
        # then we need to "select" them to make them highlighted in the listbox
        else:
            list_view = self.controls.privileged_pa_values
            model = list_view.model()
            selection_model = list_view.selectionModel()

            # Deselect all items first
            selection_model.clearSelection()

            # Select the items that match self.privileged_pa_values
            for value in self.privileged_pa_values:
                index = model.indexOf(value)
                if index != -1:  # -1 means the value was not found in the model
                    selection_model.select(
                        model.index(index), QItemSelectionModel.Select
                    )

    # Adding the protected attribute and favorable class value as attributes to the data domain
    def as_fairness_data(self, data: Table) -> Optional[Table]:
        """
        This function adds the protected attribute and favorable
        class value as attributes to the data domain.
        """
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
            # If the attribute is the protected attribute, then create a new attribute with
            # the same values but add the privileged_pa_values attribute.
            if attribute.name == self.protected_attribute.name:
                new_attr = attribute.copy()
                new_attr.attributes["privileged_pa_values"] = self.privileged_pa_values
            # Else copy the attribute and remove the privileged_pa_values attribute if it exists
            else:
                new_attr = attribute.copy()
                if "privileged_pa_values" in new_attr.attributes:
                    del new_attr.attributes["privileged_pa_values"]

            new_attributes.append(new_attr)

        # Create new class_var
        new_class_var = old_domain.class_var.copy()
        new_class_var.attributes["favorable_class_value"] = self.favorable_class_value

        new_domain = Domain(new_attributes, new_class_var, old_domain.metas)

        # Create a new table with the new domain
        new_data = data.transform(new_domain)
        return new_data

    # The defered allows us to only call the function once the user has stopped
    # changing the values of the comboboxes or listboxes and "Applies" the changes
    @gui.deferred
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
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    table = Table("https://datasets.biolab.si/core/adult.tab")
    WidgetPreview(OWAsFairness).run(input_data=table)
