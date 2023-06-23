from typing import Optional

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.data import Table

class OWAsFairness(OWWidget):
    name = "As Fairness Data Hard"
    description = "Converts a dataset to a fairness dataset with marked favorable class values, protected attributes and privileged protected attribute values."
    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    favorable_class_value = ['>50K',]
    protected_attributes = ['sex', 'race']
    privileged_PA_values = ['Male', 'White']

    favorable_class_value_index = Setting(0)
    protected_attribute_index = Setting(0)
    privileged_PA_values_indices = Setting([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data: Optional[Table] = None

        box = gui.vBox(self.controlArea, 'Favorable Class', margin=0)
        gui.comboBox(
            box,
            self,
            'favorable_class_value_index',
            items=self.favorable_class_value,
            callback=self.commit,
            searchable=True,
        )

        box = gui.vBox(self.controlArea, 'Protected Attribute', margin=0)
        gui.comboBox(
            box,
            self,
            'protected_attribute_index',
            items=self.protected_attributes,
            callback=self.commit,
            searchable=True,
        )

        box = gui.vBox(self.controlArea, 'Privileged Values', margin=0)
        gui.listBox(
            box,
            self,
            'privileged_PA_values_indices',
            # items=self.privileged_PA_values,
            callback=self.commit,
            selectionMode=gui.QListView.ExtendedSelection,
        )

    @Inputs.data
    def set_data(self, data: Table) -> None:
        self._data = None

        if data:
            # Create a table with the same data but a different domain
            self._data = data.transform(data.domain)
            # Copy the attributes from the original data to the new data
            self._data.attributes = data.attributes.copy()

        # Apply the changes and send the data to the output
        # self.commit.now()
        self.commit()

    # @gui.deferred #The deferred decorator is used to postpone the execution of the method until the widget is fully initialized or untill we call the self.commit.now().
    def commit(self) -> None:
        if self._data is not None:
            self._data.attributes['favorable_class'] = self.favorable_class_value[self.favorable_class_value_index]
            self._data.attributes['protected_attribute'] = self.protected_attributes[self.protected_attribute_index]
            self._data.attributes['privileged_values'] = [self.privileged_PA_values[i] for i in self.privileged_PA_values_indices]
            print(f"All attributes: {self._data.attributes}")
        self.Outputs.data.send(self._data)
