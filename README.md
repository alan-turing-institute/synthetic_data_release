# Privacy evaluation framework for synthetic data publishing
A practical framework to evaluate the privacy-utility tradeoff of synthetic data publishing 

Based on "Synthetic Data - Anonymisation Groundhog Day, Theresa Stadler, Bristena Oprisanu, and Carmela Troncoso, [arXiv](https://arxiv.org/abs/2011.07018), 2020"


# Models included

## Attack models
The module `attack_models` so far includes

A privacy adversary to test for privacy gain with respect to linkage attacks modelled as a membership inference attack `MIAAttackClassifier`. The classifiers available for the attack are:
- `LogisticRegression` from `sklearn`
- `RandomForestClassifier` from `sklearn`
- `KNeighborsClassifier` (k-Nearest Neighbours) from `sklearn`
- `MLPClassifier` (multilayer perceptron) from `sklearn`
- `SVC` (support vector classifier) from `sklearn`

A simple attribute inference attack `AttributeInferenceAttack` that aims to infer a target's sensitive value given partial knowledge about the target record

## Generative models
The module `generative_models` so far includes:   
- `IndependentHistogram`: An independent histogram model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `BayesianNet`: A generative model based on a Bayesian Network adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `PrivBayes`: A differentially private version of the BayesianNet model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `CTGAN`: A conditional tabular generative adversarial network that integrates the CTGAN model from [CTGAN](https://github.com/sdv-dev/CTGAN)


# Setup

## Requirements
The framework and its building blocks have been developed and tested under Python 3.9.


### Poetry installation
To mimic our environment exactly, we recommend using `poetry`. To install poetry (system-wide), follow the instructions [here](https://python-poetry.org/docs/).

Then run
```
poetry install
```
from inside the project directory. This will create a virtual environment (default `.venv`), that can be accessed by running `poetry shell`, or in the usual way with `source .venv/bin/activate`.


### Pip installation

For Pip installation, we recommend creating a virtual environment for installing all dependencies by running
```
python3 -m venv pyvenv3
source pyvenv3/bin/activate
pip install -r requirements.txt
```


# Example runs

### Membership inference
To run a privacy evaluation with respect to the privacy concern of linkability you can run
```
python3 linkage_cli.py -D data/texas -RC configs/linkage/runconfig.json -O runs/linkage
```
You can edit the configuration file `configs/linkage/runconfig.json` to choose the 
parameters of the evaluation:
- `nIter`: Number of independent iterations of the attack
- `sizeRawA`: Size of the shadow raw dataset (sample from the raw data that the intruder has access to)
- `nSynA`: Number of synthetic datasets to generate per trained generative model
- `nShadows`: Number of different instances of each generative model to train per target
- `sizeRawT`: Size of raw datasets to be used as training data for generative models
- `sizeSynT`: Size of each synthetic dataset to generate
- `nSynT`: Number of synthetic datasets to generate from each trained generative model
- `nTargets`: Number of targets to randomly select and perform linkage attack on (in *addition* to the ones specified by `Targets`)
- `Targets`: List of specific targets to perform linkage attack on. Each target is given in the format "IDXXX" where XXX is their sequential order in the raw data.
- `generativeModels`: Dictionary of synthetic methods to test (with parameters for each)
- `sanitisationTechniques`: Dictionary of traditional anonymisation (sanitisation) methods to test (with parameters for each)

The results file produced after successfully running the script will be written to `runs/linkage` and can be parsed with the function `load_results_linkage` provided in `utils/analyse_results.py`.
A jupyter notebook to visualise and analyse the results is included at `notebooks/Analyse Results.ipynb`.

### Attribute inference
To run a privacy evaluation with respect to the privacy concern of inference you can run
```
python3 inference_cli.py -D data/texas -RC configs/inference/runconfig.json -O runs/inference
```
You can edit the configuration file `configs/inference/runconfig.json` to choose the 
parameters of the evaluation:
- `sensitiveAttributes`: Dictionary of the sensitive attributes and whether they should be attacked using classification or regression
- `nIter`: Number of independent iterations of the attack
- `sizeRawT`: Size of raw datasets to be used as training data for generative models
- `sizeSynT`: Size of each synthetic dataset to generate
- `nSynT`: Number of synthetic datasets to generate from each trained generative model
- `nTargets`: Number of targets to randomly select and perform inference attack on (in *addition* to the ones specified by `Targets`)
- `Targets`: List of specific targets to perform inference attack on. Each target is given in the format "IDXXX" where XXX is their sequential order in the raw data.
- `generativeModels`: Dictionary of synthetic methods to test (with parameters for each)
- `sanitisationTechniques`: Dictionary of traditional anonymisation (sanitisation) methods to test (with parameters for each)

The results file produced after successfully running the script will be written to `runs/inference` and can be parsed with the function `load_results_inference` provided in `utils/analyse_results.py`.
A jupyter notebook to visualise and analyse the results is included at `notebooks/Analyse Results.ipynb`.

### Utility
To run a utility evaluation with respect to a simple classification task as utility function run

```
python3 utility_cli.py -D data/texas -RC configs/utility/runconfig.json -O runs/utility
```
You can edit the configuration file `configs/utility/runconfig.json` to choose the 
parameters of the evaluation:
- `nIter`: Number of independent iterations of the attack
- `sizeRawT`: Size of raw datasets to be used as training data for generative models
- `sizeSynT`: Size of each synthetic dataset to generate
- `nSynT`: Number of synthetic datasets to generate from each trained generative model
- `nTargets`: Number of targets to randomly select and evaluate utility on (in *addition* to the ones specified by `Targets`)
- `Targets`: List of specific targets to include/not include when doing utility evaluation. Each target is given in the format "IDXXX" where XXX is their sequential order in the raw data.
- `TestRecords`: List of specific targets to perform utility evaluation on. Each target is given in the format "IDXXX" where XXX is their sequential order in the raw data.
- `generativeModels`: Dictionary of synthetic methods to test (with parameters for each)
- `sanitisationTechniques`: Dictionary of traditional anonymisation (sanitisation) methods to test (with parameters for each)
- `utilityTasks`: Dictionary of classification models to train and for which label variables
- `dataFilter`: Dictionary indicating train/test split of data

The results file produced after successfully running the script will be written to `runs/utility` and can be parsed with the function `load_results_utility` provided in `utils/analyse_results.py`.
A jupyter notebook to visualise and analyse the results is included at `notebooks/Analyse Results.ipynb`.
