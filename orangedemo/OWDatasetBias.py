from typing import Optional

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel
from Orange.data import Table, Domain, DiscreteVariable

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric

class OWDatasetBias(OWWidget):
    name = "Dataset Bias"
    description = "Computes the bias of a dataset."
    # icon = "icons/bias.svg"
    # priority = 0

    standardDataset = None
    
    class Inputs:
        data = Input("Data", Table)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data: Optional[Table] = None

    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        # self.closeContext()

        if not data:
            return
        
        self._data = data
        # self.openContext(data)

        self.convertToStandardDataset(data)

    def convertToStandardDataset(self, data) -> None:
        # Convert Orange data to aif360 dataset
        df = data.to_pandas_dfs()
        favorable_class_value = data.attributes["favorable_class_value"]
        protected_attribute = data.attributes["protected_attribute"]
        privileged_PA_values = data.attributes["privileged_PA_values"]

        self.standardDataset = StandardDataset(
            df = df,
            label_name = "y",
            favorable_classes = [favorable_class_value],
            protected_attribute_names = [protected_attribute],
            privileged_classes = [privileged_PA_values]
        )
        