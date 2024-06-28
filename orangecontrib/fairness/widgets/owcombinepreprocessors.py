"""
This module contains the implementation of the Combine Preprocessors widget.

This widget allows the user to combine two preprocessors into 
one so it can be used as input for other widgets.
"""

from typing import Optional

from Orange.widgets import gui
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.preprocess.preprocess import Preprocess, PreprocessorList


class OWCombinePreprocessors(OWWidget):
    """
    Widget for combining 2 preprocessors into one so it can be used as input for other widgets.
    """

    name = "Combine Preprocessors"
    description = "Combine multiple preprocessors into one."
    icon = "icons/combine_preprocessors.svg"
    priority = 60

    want_control_area = False
    resizing_enabled = False

    class Inputs:
        """Input for the widget - two preprocessors or preprocessor lists."""

        first_preprocessor = Input("First Preprocessor", Preprocess)
        second_preprocessor = Input("Second Preprocessor", Preprocess)

    class Outputs:
        """Output for the widget - combined preprocessor."""

        preprocessor = Output("Preprocessor", Preprocess)

    def __init__(self):
        self.preprocessor_list = []
        self.first_preprocessor = None
        self.second_preprocessor = None

        super().__init__()

        box = gui.vBox(self.mainArea, "Info")
        gui.widgetLabel(
            box,
            (
                "This widgets allows you to combine two preprocessors "
                "into one and use it as input for other widgets."
            ),
        )

    @Inputs.first_preprocessor
    def set_first_preprocessor(self, preprocessor: Optional[Preprocess]) -> None:
        """Storing the preprocessor entered in the first input."""
        self.first_preprocessor = preprocessor

    @Inputs.second_preprocessor
    def set_second_preprocessor(self, preprocessor: Optional[Preprocess]) -> None:
        """Storing the preprocessor entered in the second input."""
        self.second_preprocessor = preprocessor

    def handleNewSignals(self):
        """Combining the two preprocessors into one and sending it as output."""
        self.preprocessor_list = []
        if self.first_preprocessor is not None:
            self.preprocessor_list.append(self.first_preprocessor)
        if self.second_preprocessor is not None:
            self.preprocessor_list.append(self.second_preprocessor)

        self.Outputs.preprocessor.send(PreprocessorList(self.preprocessor_list))
