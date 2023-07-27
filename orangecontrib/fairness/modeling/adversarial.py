from Orange.base import Learner, Model
from Orange.data import Table
from Orange.preprocess import Normalize

from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

from orangecontrib.fairness.widgets.utils import (
    table_to_standard_dataset,
    contains_fairness_attributes,
    MISSING_FAIRNESS_ATTRIBUTES,
)

import numpy as np


# This gets called after the model is created and fitted
# It is stored so we can use it to predict on new data
class AdversarialDebiasingModel(Model):
    def __init__(self, model, learner):
        super().__init__()
        self._model = model
        self.learner = learner
        self.params = vars()

    def predict(self, data):
        if isinstance(data, Table):
            # Normalize the data
            data = self.learner.preprocess(data)
            # For creating the standard dataset we need to know the encoding the table uses for the class variable, the encoding is ordinal and is the same as the order of values in the domain
            if not data.domain.class_var:
                data.domain.class_var = self.original_domain.class_var
            standard_dataset, _, _ = table_to_standard_dataset(data)
            predictions = self._model.predict(standard_dataset)

            # Create a array of scores with a column for each class the first column is the predictions.scores and the second column is 1 - predictions.scores
            # TODO: Check if the order of the columns is always correct
            scores = np.hstack(
                (predictions.scores, (1 - predictions.scores).reshape(-1, 1))
            )

            return np.squeeze(predictions.labels, axis=1), scores
        else:
            raise TypeError("Data is not of type Table")

    def predict_storage(self, data):
        if isinstance(data, Table):
            return self.predict(data)
        else:
            raise TypeError("Data is not of type Table")

    def __call__(self, data, ret=Model.Value):
        return self.predict_storage(data)


class AdversarialDebiasingLearner(Learner):
    __returns__ = AdversarialDebiasingModel
    # List of preprocessors, these get applied when the __call__ function is called
    preprocessors = [
        Normalize()
    ]
    callback = None

    def __init__(self, preprocessors=None, **kwargs):
        self.params = vars()
        super().__init__(preprocessors=preprocessors)

    def incompatibility_reason(self, domain):
        if not contains_fairness_attributes(domain):
            return MISSING_FAIRNESS_ATTRIBUTES

    def fit_storage(self, data):
        return self.fit(data)

    def _fit_model(self, data):
        if type(self).fit is Learner.fit:
            return self.fit_storage(data)
        else:
            return self.fit(data)

    # Function responsible for fitting the learner to the data and creating a model
    # TODO: Should I use the X,Y,W format instead of the table format ?
    def fit(self, data: Table) -> AdversarialDebiasingModel:
        standardDataset, privileged_groups, unprivileged_groups = table_to_standard_dataset(data)
        # Create a new session and reset the default graph
        # Eager execution mea
        tf.disable_eager_execution()
        tf.reset_default_graph()
        if tf.get_default_session() is not None:
            tf.get_default_session().close()
        sess = tf.Session()

        # Create a model using the parameters from the widget and fit it to the data
        # **self.params["kwargs"] unpacks the dictionary self.params["kwargs"] into keyword arguments
        model = ThreadedAdversarialDebiasing(
            **self.params["kwargs"],
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            sess=sess,
            scope_name="adversarial_debiasing"
        )
        model = model.fit(standardDataset, callback=self.callback)
        return AdversarialDebiasingModel(model=model, learner=self)

    # This is called when using the learner as a function, in the superclass it uses the _fit_model function
    # Which creates a new model by calling the fit function
    def __call__(self, data, progress_callback=None):
        self.callback = progress_callback
        model = super().__call__(data, progress_callback)
        model.params = self.params
        return model
    





# This is the same as the AdversarialDebiasing class from aif360.algorithms.inprocessing.adversarial_debiasing
# The only difference is that the fit method in this class has a callback parameter which is used to report the progress of the fitting
class ThreadedAdversarialDebiasing(AdversarialDebiasing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, dataset, callback=None):
        """Compute the model parameters of the fair classifier using gradient
        descent.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            AdversarialDebiasing: Returns self.
        """
        if tf.executing_eagerly():
            raise RuntimeError("AdversarialDebiasing does not work in eager "
                    "execution mode. To fix, add `tf.disable_eager_execution()`"
                    " to the top of the calling script.")

        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4 = np.random.randint(ii32.min, ii32.max, size=4)

        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0

        with tf.variable_scope(self.scope_name):
            num_train_samples, self.features_dim = np.shape(dataset.features)

            # Setup placeholders
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            if self.debias:
                # Obtain adversary predictions and adversary loss
                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(pred_logits, self.true_labels_ph)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph, logits=pred_protected_attributes_logits))

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       1000, 0.96, staircase=True)
            classifier_opt = tf.train.AdamOptimizer(learning_rate)
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(learning_rate)

            classifier_vars = [var for var in tf.trainable_variables(scope=self.scope_name) if 'classifier_model' in var.name]
            if self.debias:
                adversary_vars = [var for var in tf.trainable_variables(scope=self.scope_name) if 'adversary_model' in var.name]
                # Update classifier parameters
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                                      var_list=classifier_vars)}
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            classifier_grads = []
            for (grad,var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):
                if self.debias:
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]
                classifier_grads.append((grad, var))
            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                with tf.control_dependencies([classifier_minimizer]):
                    adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars)#, global_step=global_step)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # Begin training
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples//self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
                    batch_features = dataset.features[batch_ids]
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1,1])
                    batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                                 dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

                    batch_feed_dict = {self.features_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.protected_attributes_ph: batch_protected_attributes,
                                       self.keep_prob: 0.8}
                    if self.debias:
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run([classifier_minimizer,
                                       adversary_minimizer,
                                       pred_labels_loss,
                                       pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        # if i % 200 == 0:
                        #     print("epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (epoch, i, pred_labels_loss_value,
                        #                                                              pred_protected_attributes_loss_vale))
                    else:
                        _, pred_labels_loss_value = self.sess.run(
                            [classifier_minimizer,
                             pred_labels_loss], feed_dict=batch_feed_dict)
                        # if i % 200 == 0:
                        #     print("epoch %d; iter: %d; batch classifier loss: %f" % (
                        #     epoch, i, pred_labels_loss_value))
                if callback is not None:
                    progress = round(epoch / self.num_epochs * 100, 0)
                    callback(progress, "Fitting...")
        return self
    

# This is only a temporary solution, doing it this way makes me prone to errors as a result of changes in the aif360 library, to combat this I specify the exact version of aif360 in the requirements
# TODO: Find a better way to report the progress of the fitting, some ideas:
# 1. Make a subclass of the tensorflow session (sess) and override the run function so it calls the callback function
# 2. Make a monkey patch for the tensorflow session (sess) and override the run function so it calls the callback function
# To calculate the progress using these ways we need to know the number of expected calls to the callback function and count how many times it has been called
# After fixing this, I can delete the specific version of aif360 from the requirements