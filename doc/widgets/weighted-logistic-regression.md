Weighted Logistic Regression
================
Logistic regression model that supports instance weights.

**Inputs**

- Data: reference dataset
- Preprocessor: preprocessing method(s)

**Outputs**

- Learner: logistic regression learning algorithm
- Model: trained model
- Coefficients: logistic regression coefficients

**Weighted Logistic Regression** this widget is a logistic regression model that supports instance weights. It is useful when we want to predict on data preprocessed with the [Reweighing](reweighing.md) widget or use the [Reweighing](reweighing.md) widget as a preprocessor for the learnes. At the time of writing it is the only learner that supports instance weights.

Besides the support for weights it works in the exact same way as the normal [Logistic Regression](https://orange3.readthedocs.io/projects/orange-visual-programming/en/latest/widgets/model/logisticregression.html) widget so for more details check its documentation.

Example
-------

The example of how to use the `Weighted Logistic Regression` can be found on the [Reweighing](reweighing.md) widget documentation. An example of how to use the `Logistic Regression` widget in general can be found on the [Logistic Regression](https://orange3.readthedocs.io/projects/orange-visual-programming/en/latest/widgets/model/logisticregression.html) widget documentation.