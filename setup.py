import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IronDrive-stieloli",
    version="0.0.1",
    author="Oliver Roman Stiel",
    author_email="oliver.roman.stiel@effem.com",
    description="Package for handling and managing hyperparameters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
