from setuptools import setup, find_packages

setup(
    name="bdgs",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'tensorflow-cpu',
        'scikit-learn',
        'keras',
        'scikit-image',
        'silence-tensorflow'
    ],
    include_package_data=True,
    package_data={
        'trained_models': ['*'],
    },
    py_modules=['definitions'],
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
