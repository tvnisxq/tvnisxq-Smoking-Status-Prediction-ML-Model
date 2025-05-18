from setuptools import setup, find_packages

setup(
    name="SmokingML",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "seaborn",
        "matplotlib"
    ]
)