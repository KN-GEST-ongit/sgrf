# BDGS library development setup

## Prerequisities

- Python 3.11
- PIP

## Installation

1. Create/activate virtual enviroment
2. Install required packages with `pip install -r requirements.txt`

## Usage

### Create new algorithm

To create a new algorithm, use algorithm creation script: `./scripts/generate_algorithm.py`

### Import library

To use library in external project, clone this repository and use `pip install --ignore-installed ../bdgs` (pointing to local repository instance)

### Validate algorithms

To validate algorithms use scripts located in `./validation` directory. Json files with validation results are located in `./validation/results` directory
