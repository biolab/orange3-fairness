from setuptools import setup, find_packages

setup(
    name="Fairness",
    packages=find_packages(),
    package_data={
        "orangecontrib.fairness.widgets": ["icons/*"],
        },
    classifiers=["Example :: Invalid"],
    entry_points={
        "orange3.addon": ("Orange3-Fairness = orangecontrib.fairness",),
        "orange.widgets": ("Fairness = orangecontrib.fairness.widgets",),
        },
    install_requires=[
        "Orange3",
        "tensorflow>=2.12.0",
        "aif360==0.5.0",
        "numpy~=1.23.0"
    ]
)