from setuptools import find_packages, setup

setup(
    name="mlops_heart_disease",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[],
)
