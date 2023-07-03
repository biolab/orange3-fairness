from typing import Optional

from itertools import chain

from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel
from Orange.data import Table, Domain, DiscreteVariable

from AnyQt.QtWidgets import QFormLayout, QLabel
from AnyQt.QtCore import Qt, QThread, QObject

from orangedemo.modeling.adversarial import AdversarialDebiasingLearner

from Orange.widgets.utils.widgetpreview import WidgetPreview


class OWAdversarialDebiasing(OWBaseLearner):
    name = "Adversarial Debiasing"
    description = "Adversarial Debiasing classification algorithm with or without fairness constraints."
    # icon = "icons/AdversarialDebiasing.svg"
    # priority = 10

    class Inputs(OWBaseLearner.Inputs):
        pass

    class Outputs(OWBaseLearner.Outputs):
        pass

    LEARNER = AdversarialDebiasingLearner

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

    hidden_layers_neurons = Setting(100)
    number_of_epochs = Setting(50)
    batch_size = Setting(128)
    debias = Setting(True)
    lambda_index = Setting(1)

    def add_main_layout(self):
        form = QFormLayout()
        form.setFieldGrowthPolicy(form.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)
        gui.widgetBox(self.controlArea, True, orientation=form)
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
        form.addRow(
            gui.checkBox(
                None,
                self,
                "debias",
                label="Use debiasing",
                callback=self.settings_changed,
                attribute=Qt.WA_LayoutUsesWidgetRect,
            )
        )

        self.reg_label = QLabel()
        slider = gui.hSlider(
            None,
            self,
            "lambda_index",
            minValue=0,
            maxValue=len(self.lambdas) - 1,
            callback=lambda: (self.set_lambda(), self.settings_changed()),
            createLabel=False,
        )
        form.addRow(self.reg_label, slider)
        self.set_lambda()

    def set_lambda(self):
        # called from init, pylint: disable=attribute-defined-outside-init
        self.strength_C = self.lambdas[self.lambda_index]
        self.reg_label.setText("Adversary Loss Weight, λ={}:".format(self.strength_C))

    @property
    def selected_lambda(self):
        return self.lambdas[self.lambda_index]

    def setup_layout(self):
        return super().setup_layout()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.update_model()

    def update_model(self):
        super().update_model()
        if self.model is not None:
            self.Outputs.model.send(self.model)

    def create_learner(self):
        return self.LEARNER(
            classifier_num_hidden_units=self.hidden_layers_neurons,
            num_epochs=self.number_of_epochs,
            batch_size=self.batch_size,
            debias=self.debias,
            adversary_loss_weight=self.selected_lambda,
        )

    def handleNewSignals(self):
        self.apply()

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


if __name__ == "__main__":
    table = Table('http://datasets.biolab.si/core/melanoma.tab')
    WidgetPreview(OWAdversarialDebiasing).run(input_data=table)
