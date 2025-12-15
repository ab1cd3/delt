# setup.py

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = f.read()

setup(
    name="delt",
    version="0.1.0",
    description="delt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    license="MIT",

    package_dir={"": "src"},
    packages=find_packages("src"),

    install_requires=[install_requires],
    include_package_data=True,
)