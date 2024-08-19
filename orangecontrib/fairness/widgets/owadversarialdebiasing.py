"""
This module contains the OWAdversarialDebiasing widget.
"""

from itertools import chain

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.data import Table
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.base import Model
from Orange.widgets.widget import Msg

from orangecanvas.application.addons import AddonManagerDialog, Installable

from AnyQt.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QPushButton
from AnyQt.QtCore import Qt

from orangecontrib.fairness.modeling.adversarial import AdversarialDebiasingLearner
from orangecontrib.fairness.widgets.utils import (
    check_fairness_data,
    check_for_reweighing_preprocessor,
    check_for_reweighted_data,
    check_for_missing_values,
    check_for_tensorflow,
    is_tensorflow_installed,
    TENSORFLOW_NOT_INSTALLED,
)


class InterruptException(Exception):
    """A dummy exception used to interrupt the training process."""

    pass


class AdversarialDebiasingRunner:
    """
    A class used to run the AdversarialDebiasingLearner in a separate
    thread and display progress using the callback.
    """

    @staticmethod
    def run(
        learner: AdversarialDebiasingLearner, data: Table, state: TaskState
    ) -> Model:
        """
        Method used to run the AdversarialDebiasingLearner in a separate
        thread and display progress using the callback.
        """
        if data is None:
            return None

        def callback(progress: float, msg: str = None) -> bool:
            state.set_progress_value(progress)
            if state.is_interruption_requested():
                raise InterruptException

        model = learner(data, progress_callback=callback)
        return model


class OWAdversarialDebiasing(ConcurrentWidgetMixin, OWBaseLearner):
    """A widget used to customize and create the AdversarialDebiasing Learner and/or Model"""

    name = "Adversarial Debiasing"
    description = (
        "Adversarial Debiasing classification algorithm "
        "with or without fairness constraints."
    )
    icon = "icons/adversarial_debiasing.svg"
    priority = 30

    resizing_enabled = True

    class Inputs(OWBaseLearner.Inputs):
        """Inputs for the widgets, which are the same as for the super class (Data, Preprocessor)"""

        pass

    class Outputs(OWBaseLearner.Outputs):
        """Outputs for the widgets, which are the same as for the super class (Learner, Model)"""

        pass

    class Information(OWBaseLearner.Information):
        """Information shown to the user when the user specifies custom preprocessors"""

        # This was slightly changed from the original to fit the new widget better
        ignored_preprocessors = Msg(
            "Ignoring default preprocessing. \n"
            "Default preprocessing (scailing), has been "
            "replaced with user-specified preprocessors. \n"
            "Problems may occur if these are inadequate for the given data."
        )

    # We define the learner we want to use
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
    hidden_layers_neurons = Setting(100)
    number_of_epochs = Setting(50)
    batch_size = Setting(128)
    debias = Setting(True)
    lambda_index = Setting(1)
    repeatable = Setting(False)

    def __init__(self):
        ConcurrentWidgetMixin.__init__(self)
        OWBaseLearner.__init__(self)

    def tensorflow_layout(self):
        """Defines the main UI layout of the widget if the user has tensorflow installed"""
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

    def no_tensorflow_layout(self):
        """Defines the main UI layout of the widget if the user doesn't have tensorflow installed"""

        layout = QVBoxLayout()
        label = QLabel(
            "The Adversarial Debiasing widget requires TensorFlow, which is not installed.\n"
            'You can install it by clicking the "Install TensorFlow" button below, selecting \n'
            'the checkbox next to the "tensorflow" text and clicking the "Ok" button.\n'
            "After that, you will need to restart Orange."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        button = QPushButton("Install TensorFlow")
        button.clicked.connect(self.install_tensorflow)
        layout.addWidget(button)

        box = gui.widgetBox(self.controlArea, True, orientation=layout)

        self.Error.add_message("no_tensorflow", TENSORFLOW_NOT_INSTALLED)
        self.Error.no_tensorflow()

    def install_tensorflow(self):
        """
        Installs tensorflow
        """

        installable = Installable(
            name="tensorflow",
            version="2.15.0",
            summary="TensorFlow is an open source machine learning framework for everyone.",
            description="[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)\n[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)\n\nTensorFlow is an open source software library for high performance numerical\ncomputation. Its flexible architecture allows easy deployment of computation\nacross a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters\nof servers to mobile and edge devices.\n\nOriginally developed by researchers and engineers from the Google Brain team\nwithin Google's AI organization, it comes with strong support for machine\nlearning and deep learning and the flexible numerical computation core is used\nacross many other scientific domains. TensorFlow is licensed under [Apache\n2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).\n",
            package_url="https://pypi.org/project/tensorflow/",
            release_urls=[],
            requirements=[
                "absl-py (>=1.0.0)",
                "astunparse (>=1.6.0)",
                "flatbuffers (>=23.5.26)",
                "gast (!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1)",
                "google-pasta (>=0.1.1)",
                "h5py (>=2.9.0)",
                "libclang (>=13.0.0)",
                "ml-dtypes (~=0.2.0)",
                "numpy (<2.0.0,>=1.23.5)",
                "opt-einsum (>=2.3.2)",
                "packaging",
                "protobuf (!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3)",
                "setuptools",
                "six (>=1.12.0)",
                "termcolor (>=1.1.0)",
                "typing-extensions (>=3.6.6)",
                "wrapt (<1.15,>=1.11.0)",
                "tensorflow-io-gcs-filesystem (>=0.23.1)",
                "grpcio (<2.0,>=1.24.3)",
                "tensorboard (<2.16,>=2.15)",
                "tensorflow-estimator (<2.16,>=2.15.0)",
                "keras (<2.16,>=2.15.0)",
                "nvidia-cublas-cu12 (==12.2.5.6) ; extra == 'and-cuda'",
                "nvidia-cuda-cupti-cu12 (==12.2.142) ; extra == 'and-cuda'",
                "nvidia-cuda-nvcc-cu12 (==12.2.140) ; extra == 'and-cuda'",
                "nvidia-cuda-nvrtc-cu12 (==12.2.140) ; extra == 'and-cuda'",
                "nvidia-cuda-runtime-cu12 (==12.2.140) ; extra == 'and-cuda'",
                "nvidia-cudnn-cu12 (==8.9.4.25) ; extra == 'and-cuda'",
                "nvidia-cufft-cu12 (==11.0.8.103) ; extra == 'and-cuda'",
                "nvidia-curand-cu12 (==10.3.3.141) ; extra == 'and-cuda'",
                "nvidia-cusolver-cu12 (==11.5.2.141) ; extra == 'and-cuda'",
                "nvidia-cusparse-cu12 (==12.1.2.141) ; extra == 'and-cuda'",
                "nvidia-nccl-cu12 (==2.16.5) ; extra == 'and-cuda'",
                "nvidia-nvjitlink-cu12 (==12.2.140) ; extra == 'and-cuda'",
                "tensorrt (==8.6.1.post1) ; extra == 'and-cuda'",
                "tensorrt-bindings (==8.6.1) ; extra == 'and-cuda'",
                "tensorrt-libs (==8.6.1) ; extra == 'and-cuda'",
            ],
            description_content_type="text/markdown",
        )

        manager = AddonManagerDialog(self)
        manager.runQueryAndAddResults([])
        manager.addInstallable(installable)

    def add_main_layout(self):
        if is_tensorflow_installed():
            self.tensorflow_layout()
        else:
            self.no_tensorflow_layout()

    # ---------Methods related to UI------------

    def set_lambda(self):
        """Responsible for the text of the adversary loss weight slider"""
        self.strength_D = self.lambdas[self.lambda_index]
        self.reg_label.setText(f"Adversary Loss Weight, λ={self.strength_D}:")

    @property
    def selected_lambda(self):
        """Responsible for getting the lambda value from the index of the slider value"""
        return self.lambdas[self.lambda_index]

    def _debias_changed(self):
        """Responsible for enabling/disabling the slider"""
        self.slider.setEnabled(self.debias)
        self.apply()

    # ---------Methods related to inputs--------------

    @Inputs.data
    @check_for_tensorflow
    @check_fairness_data
    @check_for_reweighted_data
    @check_for_missing_values
    def set_data(self, data):
        """
        Function which is called when the user inputs data, it first checks if the
        data is valid, cancels the current task and then calls the superclass method.
        """
        self.cancel()
        super().set_data(data)

    @Inputs.preprocessor
    @check_for_tensorflow
    @check_for_reweighing_preprocessor
    def set_preprocessor(self, preprocessor):
        """
        Function which is called when the user inputs a preprocessor, it first checks if the
        preprocessor is valid, cancels the current task and then calls the superclass method.
        """
        self.cancel()
        super().set_preprocessor(preprocessor)

    # ----------Methods related to the learner/model--------------

    def create_learner(self):
        """
        Responsible for creating the learner with the parameters we want
        It is called in the superclass by the update_learner method
        """
        if is_tensorflow_installed():
            return self.LEARNER(
                preprocessors=self.preprocessors,
                seed=42 if self.repeatable else -1,
                classifier_num_hidden_units=self.hidden_layers_neurons,
                num_epochs=self.number_of_epochs,
                batch_size=self.batch_size,
                debias=self.debias,
                adversary_loss_weight=self.selected_lambda if self.debias else 0,
            )

    def update_model(self):
        """
        Responsible for starting a new thread, fitting the
        learner and sending the created model to the output

        This method is called along with the update_learner
        method in the apply method of the superclass
        """

        self.cancel()
        if self.data is not None:
            self.start(AdversarialDebiasingRunner.run, self.learner, self.data)
        else:
            self.Outputs.model.send(None)

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

    # -----------Methods related to threading and output-------------

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
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWAdversarialDebiasing).run()
