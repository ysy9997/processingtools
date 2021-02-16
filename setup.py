import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ProcessTools", # Replace with your own username
    version="0.0.1",
    author="syy",
    author_email="ysy9997@gmail.com",
    description="Various function for image process",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ysy9997/functions.gitt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
