from Orange.base import Learner
from Orange.data import Table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.widget import Input
from Orange.widgets import gui
from Orange.base import Model

from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

from orangecontrib.fairness.modeling.postprocessing import PostprocessingLearner
from orangecontrib.fairness.widgets.utils import check_fairness_data, check_for_fairness_learner_or_preprocessor


class InterruptException(Exception):
    pass

class EqualizedOddsRunner:
    @staticmethod
    def run(
        learner: Learner, data: Table, state: TaskState
    ) -> Model:
        if data is None:
            return None

        def callback(i: float, msg: str = None) -> bool:
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        state.set_status("Training model...")
        model = learner(data, progress_callback=callback)
        return model




class OWEqualizedOdds(ConcurrentWidgetMixin, OWBaseLearner):
    name = "Equalized Odds Postprocessing"
    description = "Postprocessing fairness algorithm which changes the predictions of a classifier to satisfy equalized odds."
    icon = "icons/eq_odds_postprocessing.svg"
    priority = 40

    LEARNER = PostprocessingLearner
    repeatable = Setting(True)

    class Inputs(OWBaseLearner.Inputs):
        learner = Input("Learner", Learner)

    def __init__(self):
        self.normal_learner: Learner = None
        ConcurrentWidgetMixin.__init__(self)
        OWBaseLearner.__init__(self)

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

    @Inputs.data
    @check_fairness_data
    def set_data(self, data: Table):
        self.cancel()
        super().set_data(data)


    @Inputs.learner
    @check_for_fairness_learner_or_preprocessor
    def set_learner(self, learner: Learner):
        self.cancel()
        self.normal_learner = learner
        if learner is not None:
            self.learner_name = f"Equalized Odds: {learner.name}"

    @Inputs.preprocessor
    @check_for_fairness_learner_or_preprocessor
    def set_preprocessor(self, preprocessor):
        self.cancel()
        super().set_preprocessor(preprocessor)


    def create_learner(self):
        if not self.normal_learner:
            return None
        return self.LEARNER(
            self.normal_learner,
            preprocessors=self.preprocessors,
            repeatable=self.repeatable,
        )
    
    def handleNewSignals(self):
        if not self.normal_learner:
            return
        self.update_learner() 
        if self.data is not None:
            self.update_model()

    def update_model(self):
        self.cancel()
        if self.data is not None and self.normal_learner is not None:
            self.start(EqualizedOddsRunner.run, self.learner, self.data)
        else:
            self.Outputs.model.send(None)

    def on_partial_result(self, _):
        pass

    def on_done(self, result: Model):
        assert isinstance(result, Model) or result is None
        self.model = result
        self.Outputs.model.send(result)

    def on_exception(self, ex):
        raise ex

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


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
