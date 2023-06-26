from typing import Optional

from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel
from Orange.data import Table, Domain, ContinuousVariable

class OWAsFairness(OWWidget):
    # Define the name and other details of the widget
    name = "As Fairness Data"
    description = "Converts a dataset to a fairness dataset with marked favorable class values, protected attributes and priviliged protected attribute values."
    # icon = 'icons/owassurvivaldata.svg'
    # priority = 0
    want_main_area = False
    resizing_enabled = False

    # Define inputs and outputs
    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)


    #Settings: The favorable_class, protected_attribute, and privileged_values are instance variables that are declared as ContextSetting. A ContextSetting is a special type of Setting that Orange remembers for each different context (i.e., input data domain).
    settingsHandler = DomainContextHandler()
    protected_attribute = ContextSetting(None, schema_only=True) #schema_only -> The setting is saved within the workflow but the default never changes.
    favorable_class_value = ContextSetting("", schema_only=True)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data: Optional[Table] = None

        protected_attribute_model = DomainModel(valid_types=(ContinuousVariable,)) #DomainModel stores the domain of the input data and allows us to select a attribute from it. It is tied to the gui.comboBox widget. And Can be accesed by self.controls.protected_attribute.model()
        favorable_class_items_model = PyListModel(iterable=[]) #Here we don't want to display the different attributes/features but the values of the attribute so I couldn't use the DomainModel which stores the features, so I used the PyListModel which can store any type in an iterable

        # Create a box for each of the three variables and populate them with comboboxes and listboxes.  
        box = gui.vBox(self.controlArea, 'Favorable Class Value', margin=0)
        gui.comboBox(box, self, 'favorable_class_value', model=favorable_class_items_model, callback=self.commit, searchable=True,)

        box = gui.vBox(self.controlArea, 'Protected Attribute', margin=0)
        gui.comboBox(box, self, 'protected_attribute', model=protected_attribute_model, callback=self.commit, searchable=True,)



    @Inputs.data
    def set_data(self, data: Table) -> None:
        self.closeContext()
        self._data = None
        domain: Optional[Domain] = None

        if data:
            # Create a table with the same data but a different domain
            self._data = data.transform(data.domain)
            # Copy the attributes from the original data to the new data
            self._data.attributes = data.attributes.copy()

            # A new Domain object is created from the attributes and class variables of the input data. 
            # This Domain object represents the structure of the input data (i.e., the variables it contains).
            domain = Domain(data.domain.attributes + data.domain.class_vars)

            # Changes the domain of the comboboxes to the new domain, thus updating the list of variables that can be selected.
            self.controls.protected_attribute.model().set_domain(domain)
    
            # Fill the favorable_class_items_model PyListModel with the values of the class variable of the input data.
            for value in data.domain.class_vars[0].values:
                self.controls.favorable_class_value.model().append(value)


            # Set the selected variables to the first variable in the list of variables (in the comboboxes or listboxes)
            self.protected_attribute = self.controls.protected_attribute.model()[0] if len(self.controls.protected_attribute.model()) else None
            self.favorable_class_value = self.controls.favorable_class_value.model()[0] if len(self.controls.favorable_class_value.model()) else None

            # If atleast one value for each variable is selected, then open the context for the widget using the new domain
            # This means that these settings will be remembered the next time the widget receives the same input data.
            if self.protected_attribute is not None and self.favorable_class_value is not None:
                self.openContext(domain)

        # Apply the changes and send the data to the output
        self.commit()

    # This function is called when the user changes the value of the comboboxes or listboxes
    # It changes the data attributes and outputs the data
    def commit(self) -> None:
        if self._data is not None:
            self._data.attributes['favorable_class_value'] = str(self.favorable_class_value)
            self._data.attributes['protected_attribute'] = str(self.protected_attribute)

            # print(self._data.attributes['favorable_class_value'])
            # print(self._data.attributes['protected_attribute'])
        
        self.Outputs.data.send(self._data)


        