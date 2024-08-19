<h1 align="center">Orange3-Fairness</h1>


<p align="center">
<a href="https://github.com/biolab//orange3-fairness/actions"><img alt="Actions Status" src="https://github.com/biolab/orange3-fairness/actions/workflows/test.yml/badge.svg"></a>
<a href="https://pypi.org/project/Orange3-Fairness/"><img alt="PyPI" src="https://img.shields.io/pypi/v/orange3-fairness?color=blue"></a>
<a href="https://codecov.io/gh/ZanMervic/orange3-fairness" ><img src="https://codecov.io/gh/ZanMervic/orange3-fairness/graph/badge.svg?token=MSQ0ZUPA6B"/></a>
<a href='https://orange3-fairness.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/orange3-fairness/badge/?version=latest' alt='Documentation Status' /></a>
</p>

![Example Workflow](doc/readme-screenshot.png)

Orange3 Fairness is an add-on for the [Orange3](http://orangedatamining.com/) data mining suite. 
It provides extensions for fairness-aware AI, which includes algorithms for detecting and mitigating 
different types of biases in the data and the predictions of machine learning models. 


# Easy installation

1. [Download](https://orangedatamining.com/download/) the latest Orange release from
our website.
2. Install the the fairness add-on: head to
`Options -> Add-ons...` in the menu bar. From the list of add-ons, select Fairness and confirm the installation.

# For developers


If you would like to install from cloned git repository, run

    pip install .

To register this add-on with Orange, but keep the code in the development directory
(do not copy it to Python's site-packages directory), run

    pip install -e .


###  Usage

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python -m Orange.canvas

The new widget appears in the toolbox bar under the section Fairness.
Starting up for the first time may take a while.
