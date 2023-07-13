from setuptools import setup

setup(
    name="Demo",
    packages=["orangedemo"],
    package_data={"orangedemo": ["icons/*.svg"]},
    classifiers=["Example :: Invalid"],
    # Declare orangedemo package to contain widgets for the "Demo" category
    entry_points={"orange.widgets": "Demo = orangedemo"},
    install_requires=[
        "tensorflow>=2.12.0",
        "aif360>=0.5.0",
    ]
)