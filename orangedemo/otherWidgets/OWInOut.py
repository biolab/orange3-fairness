from Orange import data
from Orange.widgets.widget import OWWidget, Input, Output

class InOut(OWWidget):
    name = "InOut"
    description = "Output the input data"

    class Inputs:
        data = Input("Data", data.Table)
    
    class Outputs:
        data = Output("Data", data.Table)

    @Inputs.data
    def set_data(self, data):
        self.Outputs.data.send(data)

    def __init__(self):
        super().__init__()

