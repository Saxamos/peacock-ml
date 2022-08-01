from setuptools import setup

setup(
    name="peacock-ml",
    long_description=open("README.md").read(),
    packages=["peacock_ml"],
    install_requires=["sklearn", "xgboost", "pytest", "pandas"],
)
