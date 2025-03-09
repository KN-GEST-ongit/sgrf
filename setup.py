from setuptools import setup, find_packages

setup(
    name="bdgs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="GEST science club, Rzeszów University of Technology",
    author_email="rut-ai@kia.prz.edu.pl",
    description="Static gestures recognition tool",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
