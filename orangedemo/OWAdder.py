from Orange.widgets.widget import OWWidget, Input, Output

class Adder(OWWidget):
    name = "Adder"
    description = "Add two numbers"

    class Inputs:
        number1 = Input("Number 1", int)
        number2 = Input("Number 2", int)

    class Outputs:
        sum = Output("Sum", int)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.number1 = None
        self.number2 = None

    @Inputs.number1
    def set_number1(self, number1):
        self.number1 = number1

    @Inputs.number2
    def set_number2(self, number2):
        self.number2 = number2

    def handleNewSignals(self):
        if self.number1 is not None and self.number2 is not None:
            self.Outputs.sum.send(self.number1 + self.number2)
        else:
            self.Outputs.sum.send(None)

    
    