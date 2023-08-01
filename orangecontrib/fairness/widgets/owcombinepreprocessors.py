from typing import Optional

from Orange.widgets import gui
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.preprocess.preprocess import Preprocess, PreprocessorList


class OWCombinePreprocessors(OWWidget):
    name = "Combine Preprocessors"
    description = "Combine multiple preprocessors into one."
    icon = "icons/combine_preprocessors.svg"
    priority = 60

    want_control_area = False
    resizing_enabled = False

    class Inputs:
        first_preprocessor = Input("First Preprocessor", Preprocess)
        second_preprocessor = Input("Second Preprocessor", Preprocess)

    class Outputs:
        preprocessor = Output("Preprocessor", Preprocess)

    def __init__(self):
        self.preprocessor_list = []
        self.first_preprocessor = None
        self.second_preprocessor = None
        
        super().__init__()

        box = gui.vBox(self.mainArea, "Info")
        gui.widgetLabel(
            box,
            "This widgets allows you to combine two preprocessors into one and use it as input for other widgets.",
        )


    @Inputs.first_preprocessor
    def set_first_preprocessor(self, preprocessor: Optional[Preprocess]) -> None:
        self.first_preprocessor = preprocessor

    @Inputs.second_preprocessor
    def set_second_preprocessor(self, preprocessor: Optional[Preprocess]) -> None:
        self.second_preprocessor = preprocessor
        

    def handleNewSignals(self):
        self.preprocessor_list = []
        if self.first_preprocessor is not None:
            self.preprocessor_list.append(self.first_preprocessor)
        if self.second_preprocessor is not None:
            self.preprocessor_list.append(self.second_preprocessor)

        self.Outputs.preprocessor.send(PreprocessorList(self.preprocessor_list))
        



