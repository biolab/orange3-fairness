<h1 align="center">Orange3-Fairness</h1>


<p align="center">
<a href="https://github.com/ZanMervic//orange3-fairness/actions"><img alt="Actions Status" src="https://github.com/ZanMervic/orange3-fairness/actions/workflows/test.yml/badge.svg"></a>
<a href="https://pypi.org/project/Orange3-Fairness/"><img alt="PyPI" src="https://img.shields.io/pypi/v/orange3-fairness?color=blue"></a>
<a href="https://app.codecov.io/gh/ZanMervic/orange3-fairness/"><img alt="Coverage Status" src="https://codecov.io/gh/ZanMervic/orange3-fairness/branch/main/graph/badge.svg"></a>
</p>

![Example Workflow](doc/readme-screenshot.png)

Fairness add-on for the [Orange](http://orange.biolab.si).


# Easy installation

First, [download](https://orange.biolab.si/download) the latest Orange release from
our website. Then, to install the fairness add-on, head to
`Options -> Add-ons...` in the menu bar.

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