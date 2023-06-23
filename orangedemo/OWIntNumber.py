from Orange.widgets.widget import OWWidget, Output
from Orange.widgets.settings import Setting
from Orange.widgets import gui

class IntNumber(OWWidget):
    name = "Integer Number"
    description = "Output an integer number"

    want_main_area = False
    # With a fixed non resizable geometry.
    resizing_enabled = False

    class Outputs:
        number = Output("Number", int)

    number = Setting(42)

    def __init__(self):
        super().__init__()

        from AnyQt.QtGui import QIntValidator
        gui.lineEdit(self.controlArea, self, "number", "Enter a number",
                    box="Number",
                    callback=self.number_changed,
                    valueType=int, validator=QIntValidator())
        self.number_changed()

    def number_changed(self):
        # Send the entered number on "Number" output
        self.Outputs.number.send(self.number)
