import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ProcessingTools",
    version="0.1.8",
    install_requires=['opencv-contrib-python>=3.4.10.37',
                      'numpy>=1.19.4',
                      'matplotlib>=3.3.3'],
    license='MIT',
    author="syy",
    author_email="ysy9997@gmail.com",
    description="https://github.com/ysy9997/ProcessingTools.git",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ysy9997/ProcessingTools.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
