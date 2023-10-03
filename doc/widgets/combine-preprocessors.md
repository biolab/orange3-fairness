Combine Preprocessors
================
Combines multiple preprocessors into one.

**Inputs**

- First Preprocessor: first preprocessor to be used
- Second Preprocessor: second preprocessor to be used

**Outputs**

- Preprocessor: a combination of first and second preprocessor

**Combine Preprocessors** is a widget that combines multiple preprocessors into one which can be inputed into a learner widget. This comes in handy when we want to use the reweighing widget along with some other preprocessing method. We can simply connect the reweighing widget to the first preprocessor input and the other preprocessing method to the second preprocessor input. The output of the combine preprocessors widget can then be connected to the learner widget.

Example
-------

An example of using this widget can be seen in the [Reweighing](reweighing.md) widget documentation.