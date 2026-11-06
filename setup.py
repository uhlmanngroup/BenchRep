"""Keep the repository's browsable examples importable as benchrep.examples."""
from setuptools import find_packages, setup

setup(
    packages=find_packages("src") + ["benchrep.examples", "benchrep.examples.usage"],
)
