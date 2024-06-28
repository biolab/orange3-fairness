from setuptools import setup, find_packages
from os import path, walk

VERSION = "0.2.1"

try:
    LONG_DESCRIPTION = open(
        path.join(path.dirname(__file__), "README.pypi"), "r", encoding="utf-8"
    ).read()
except FileNotFoundError:
    LONG_DESCRIPTION = ""


DATA_FILES = []


def include_documentation(local_dir, install_dir):
    global DATA_FILES
    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append((dirpath.replace(local_dir, install_dir),
                          [path.join(dirpath, f) for f in files]))
    DATA_FILES.extend(doc_files)


include_documentation('doc/_build/html', 'help/orange3-fairness')

setup(
    name="Orange3-Fairness",
    version=VERSION,
    author="Bioinformatics Laboratory, FRI UL",
    author_email="info@biolab.si",
    maintainer="Zan Mervic",
    url="https://github.com/biolab/orange3-fairness",
    description="Orange3 add-on for fairness-aware machine learning.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license="GPL3+",
    keywords=[
        "orange3 add-on",
        "orange3 fairness",
    ],
    packages=find_packages(),
    package_data={
        "orangecontrib.fairness.widgets": ["icons/*"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
    ],
    entry_points={
        "orange3.addon": ("Orange3-Fairness = orangecontrib.fairness",),
        "orange.widgets": ("Fairness = orangecontrib.fairness.widgets",),
        "orange.canvas.help": (
            'html-index = orangecontrib.fairness.widgets:WIDGET_HELP_PATH',
        ),
    },
    install_requires=[
        "numpy",
        "Orange3",
        "aif360>=0.6.0",
    ],
    extras_require={
        "doc": [
            "sphinx",
            "sphinx_rtd_theme",
            "recommonmark",
        ]
    },
    data_files=DATA_FILES,
)
