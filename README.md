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


# Installation

1. [Download](https://orangedatamining.com/download/) the latest Orange release from
our website.
2. Install the the fairness add-on: head to
`Options -> Add-ons...` in the menu bar. From the list of add-ons, select Fairness and confirm.
This will downlaod and install the add-on and its dependencies.

#  Usage

After the installation, the widget from this add-on is registered with Orange. To use it, run Orange.
The new widget appears in the toolbox bar under the section Fairness.

For an introduction to this add-on, see the following YouTube playlist:

* [Intro to Data Science](https://www.youtube.com/watch?v=H1ibqB_cvlE&list=PLmNPvQr9Tf-b_SuBdoRsuNhTmaHJ0eKab) - introduces data analysis with Orange

For more, see the following:

* [Orange widget catalog](https://orange.biolab.si/toolbox/) - Orange widgets documentation
* [Fairness add-on documentation](https://orange3-fairness.readthedocs.io/) -  documentation for Fairness widgets
* [Orange blog series on Fairness](https://orangedatamining.com/blog/?tag=fairness)


# For developers

## Installing from downloaded code

If you would like to install from cloned git repository, run

    pip install .

## Installing in editable mode

To register this add-on with Orange, but keep the code in the development directory
(do not copy it to Python's site-packages directory), run

    pip install -e .


To run Orange from the terminal, use

    orange-canvas

or

    python -m Orange.canvas
