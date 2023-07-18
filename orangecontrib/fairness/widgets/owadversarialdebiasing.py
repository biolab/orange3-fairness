from itertools import chain

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.data import Table
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.base import Model

from AnyQt.QtWidgets import QFormLayout, QLabel
from AnyQt.QtCore import Qt

from orangecontrib.fairness.modeling.adversarial import AdversarialDebiasingLearner
from orangecontrib.fairness.widgets.utils import check_fairness_data


class OWAdversarialDebiasing(ConcurrentWidgetMixin, OWBaseLearner):
    name = "Adversarial Debiasing"
    description = "Adversarial Debiasing classification algorithm with or without fairness constraints."
    # icon = "icons/AdversarialDebiasing.svg"
    # priority = 10

    # For inputs and outputs we use the same as in OWBaseLearner superclass
    class Inputs(OWBaseLearner.Inputs):
        pass

    class Outputs(OWBaseLearner.Outputs):
        pass

    # Here we define the learner we want to use, in this case it is the AdversarialDebiasingLearner
    LEARNER = AdversarialDebiasingLearner

    # We define the list of lambdas for the slider
    lambdas = list(
        chain(
            [0],
            [x / 10000 for x in range(1, 10)],
            [x / 1000 for x in range(1, 10)],
            [x / 100 for x in range(1, 10)],
            [x / 10 for x in range(1, 10)],
            range(1, 10),
            range(10, 100, 5),
            range(100, 200, 10),
            range(100, 1001, 50),
        )
    )

    # We define the settings we want to use
    # TODO: Should i use context/domain settings?
    hidden_layers_neurons = Setting(100)
    number_of_epochs = Setting(50)
    batch_size = Setting(128)
    debias = Setting(True)
    lambda_index = Setting(1)
    repeatable = Setting(False)

    def __init__(self):
        ConcurrentWidgetMixin.__init__(self)
        OWBaseLearner.__init__(self)

    # We define the UI for the widget
    def add_main_layout(self):
        form = QFormLayout()
        form.setFieldGrowthPolicy(form.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)

        gui.widgetBox(self.controlArea, True, orientation=form)
        # Spin box for the number of neurons in hidden layers
        form.addRow(
            "Neurons in hidden layers:",
            gui.spin(
                None,
                self,
                "hidden_layers_neurons",
                1,
                1000,
                step=5,
                label="Neurons in hidden layers:",
                orientation=Qt.Horizontal,
                alignment=Qt.AlignRight,
                callback=self.settings_changed,
            ),
        )
        # Spin box for the number of epochs
        form.addRow(
            "Number of epochs:",
            gui.spin(
                None,
                self,
                "number_of_epochs",
                1,
                1000,
                step=5,
                label="Number of epochs:",
                orientation=Qt.Horizontal,
                alignment=Qt.AlignRight,
                callback=self.settings_changed,
            ),
        )
        # Spin box for the batch size
        form.addRow(
            "Batch size:",
            gui.spin(
                None,
                self,
                "batch_size",
                1,
                10000,
                step=16,
                label="Batch size:",
                orientation=Qt.Horizontal,
                alignment=Qt.AlignRight,
                callback=self.settings_changed,
            ),
        )
        # Checkbox for the debiasing
        form.addRow(
            gui.checkBox(
                None,
                self,
                "debias",
                label="Use debiasing",
                callback=[self.settings_changed, self._debias_changed],
                attribute=Qt.WA_LayoutUsesWidgetRect,
            )
        )
        # Slider for the lambda (adversary loss weight)
        self.reg_label = QLabel()
        self.slider = gui.hSlider(
            None,
            self,
            "lambda_index",
            minValue=0,
            maxValue=len(self.lambdas) - 1,
            callback=lambda: (self.set_lambda(), self.settings_changed()),
            createLabel=False,
        )
        form.addRow(self.reg_label)
        form.addRow(self.slider)
        # form.addRow(self.reg_label, self.slider)
        # Checkbox for the replicable training
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
        self.set_lambda()
        self._debias_changed()

    # Responsible for the text of the lambda slider
    def set_lambda(self):
        self.strength_D = self.lambdas[self.lambda_index]
        self.reg_label.setText("Adversary Loss Weight, λ={}:".format(self.strength_D))

    # Responsible for getting the lambda value from the index of the slider
    @property
    def selected_lambda(self):
        return self.lambdas[self.lambda_index]

    # Responsible for creating the learner with the parameters we want
    # It is called in the superclass by the update_learner method
    def create_learner(self):
        kwargs = {
            "classifier_num_hidden_units": self.hidden_layers_neurons,
            "num_epochs": self.number_of_epochs,
            "batch_size": self.batch_size,
            "debias": self.debias,
            "adversary_loss_weight": 0,
        }
        if self.repeatable:
            kwargs["seed"] = 42
        if self.debias:
            kwargs["adversary_loss_weight"] = self.selected_lambda
        return self.LEARNER(**kwargs)

    def handleNewSignals(self):
        self.apply()  # This calls the update_learner and update_model methods

    # Responsible for enabling/disabling the slider
    def _debias_changed(self):
        self.slider.setEnabled(self.debias)
        self.apply()

    def get_learner_parameters(self):
        parameters = [
            ("Neurons in hidden layers", self.hidden_layers_neurons),
            ("Number of epochs", self.number_of_epochs),
            ("Batch size", self.batch_size),
            ("Use debiasing", self.debias),
        ]

        if self.debias:
            parameters.append(("Adversary Loss Weight", self.selected_lambda))

        return parameters

    @Inputs.data
    @check_fairness_data
    def set_data(self, data):
        self.cancel()
        self.data = data
        self.update_model()

    # @Inputs.preprocessor
    # def set_preprocessor(self, preprocessor):
    #     self.preprocessor = preprocessor
    #     self.update_model()

    # This method is called when the input data is changed
    # it is responsible for fitting the learner and sending the created model to the output
    # There is also a update_learner method which is called in the apply method of the superclass (along with update_model)
    def update_model(self):
        self.cancel()
        # This method will run the run_task method in a separate thread and pass the learner and data as arguments
        if self.data is not None:
            self.start(self.run_task, self.create_learner(), self.data)
        else:
            self.Outputs.model.send(None)

    def run_task(
        self, learner: AdversarialDebiasingLearner, data: Table, state: TaskState
    ) -> Model:
        model = learner(data)
        return model

    def on_partial_result(self, _):
        pass

    def on_done(self, result: Model):
        assert isinstance(result, Model) or result is None
        self.Outputs.model.send(result)

    def on_exception(self, ex):
        raise ex

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    table = Table("http://datasets.biolab.si/core/melanoma.tab")
    WidgetPreview(OWAdversarialDebiasing).run(input_data=table)
