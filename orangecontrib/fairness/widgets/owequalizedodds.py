from Orange.base import Learner
from Orange.data import Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.widget import Input
from Orange.widgets import gui

from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

from orangecontrib.fairness.modeling.postprocessing import PostprocessingLearner


class OWEqualizedOdds(OWBaseLearner):
    name = "Equalized Odds Postprocessing"
    description = "Postprocessing fairness algorithm which changes the predictions of a classifier to satisfy equalized odds."
    icon = "icons/eq_odds_postprocessing.svg"
    # priority = 10

    LEARNER = PostprocessingLearner
    repeatable = Setting(True)

    class Inputs(OWBaseLearner.Inputs):
        learner = Input("Learner", Learner)

    def __init__(self):
        self.normal_learner: Learner = None
        super().__init__()

    def add_main_layout(self):
        form = QFormLayout()
        form.setFieldGrowthPolicy(form.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)
        gui.widgetBox(self.controlArea, True, orientation=form)
        form.addRow(
            gui.checkBox(
                None,
                self,
                "repeatable",
                label="Replicable training",
                callback=self.settings_changed,
                attribute=Qt.WA_LayoutUsesWidgetRect,
            )
        )

    @Inputs.learner
    def set_learner(self, learner: Learner):
        self.normal_learner = learner
        self.handleNewSignals()

    def create_learner(self):
        if not self.normal_learner:
            return None
        return self.LEARNER(
            self.normal_learner,
            preprocessors=self.preprocessors,
            repeatable=self.repeatable,
        )


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication
    from Orange.classification.logistic_regression import LogisticRegressionLearner
    from Orange.widgets.evaluate.owpredictions import OWPredictions
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWEqualizedOdds).run()

    a = QApplication(sys.argv)
    table = Table("orangedemo/tests/datasets/adult_race_all.pkl")
    widget = OWEqualizedOdds()
    widget.set_data(table)
    widget.set_learner(LogisticRegressionLearner())
    learner = widget.create_learner()
    model = learner(table)

    predictions_widget = OWPredictions()
    predictions_widget.set_data(table)
    predictions_widget.insert_predictor(0, model)
    predictions_widget.handleNewSignals()
