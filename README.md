# Privacy evaluation framework for synthetic data publishing (UCLH CCHIC fork)

A practical framework to evaluate the privacy-utility tradeoff of synthetic data publishing.

This fork has been adapted to work within the UCLH Data Safe Haven 
([DSH](https://www.ucl.ac.uk/isd/services/file-storage-sharing/data-safe-haven-dsh)) 
environment in order to generate synthetic data that model the ICU data within the Critical Care Health 
Informatics Collaborative (CCHIC) resource. 

CCHIC is an multi-centre intensive care database in the UK (details can be found 
[here](https://discovery.ucl.ac.uk/id/eprint/10050778/)). This repository is the result of a collaboration between 
the CCHIC team in UCLH and The Alan Turing Institute 
(working under the [QUIPP](see [this](https://github.com/alan-turing-institute/QUIPP-CC-HIC) project), 
aiming to improve the CCHIC service by exploring new ways that data releases can happen, 
specifically via synthetic data, including an exploration of the tradeoffs between utility and privacy.

The code has largely been developed in the parent repository by the authors of the paper 
"Synthetic Data - Anonymisation Groundhog Day, Theresa Stadler, Bristena Oprisanu, and Carmela Troncoso, [arXiv](https://arxiv.org/abs/2011.07018), 2020".
This repository contain limited changes to the framework to allow it to run within DSH and to model the types of attacks and 
intruder assumptions which are interesting from the data owners' perspective.

# Attack models
The framework implements two type of intruder attack models which assume a motivated intruder with specific prior knowledge 
and assess their probability of success when given a synthetic (or sanitised or raw) dataset:
- A linkage attack modelled as a membership inference attack (`MIAAttackClassifier` in the code). It assumes the intruder has 
  access to a sample from the raw dataset, a set of target records and a synthetic (or other anonymised dataset) and attempts 
  to infer if each target record is part of the original raw data.
- A simple attribute inference attack (`AttributeInferenceAttack` in the code). It aims to infer a target's sensitive 
  value given partial knowledge about the target records and access to a synthetic (or other anonymised) dataset.

# Generative models
The module `generative_models` so far includes the following synthetic methods:   
- `IndependentHistogramModel`: An independent histogram model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `BayesianNetModel`: A generative model based on a Bayesian Network adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `GaussianMixtureModel`: A simple Gaussian Mixture model taken from the [sklearn library](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- `CTGAN`: A conditional tabular generative adversarial network that integrates the CTGAN model from [CTGAN](https://github.com/sdv-dev/CTGAN)  
- `PATE-GAN`: A differentially private generative adversarial network adapted from its original [implementation](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/)
- `PrivBayes`

For use within CCHIC DSH, only the `IndependentHistogramModel` and `PrivBayes` have been tested so far.

# Setup
## Requirements
The framework and its building blocks have been developed and tested under Python 3.6 and 3.7

We recommend to create a virtual environment for installing all dependencies and running the code in your
Windows DSH Anaconda Prompt:
```
python3 -m venv pyvenv3
source pyvenv3/bin/activate
pip install -r requirements.txt
```

## Dependencies
The `CTGAN` model depends on a fork of the original model training algorithm that can be found here
[CTGAN-Turing](https://github.com/alan-turing-institute/CTGAN). Note this is a version specifically created for 
use with CCHIC DSH.

To install the correct version clone the repository above and run
```
cd CTGAN
make install
```
To test your installation try to run
```
import ctgan
```
from within your virtualenv `python`.

You then need to add the path to this directory to your python path by typing the following line 
in an Anaconda Prompt window within the Windows DSH environment (note this will only work for the 
current command window session so you need to do it every time you open a new command window):
```
set PYTHONPATH=%PYTHONPATH%;ctgan_path
```
where `ctgan_path` is the path to this directory within DSH (e.g. `N:\projects\CTGAN`).

For Linux/Mac environments, you can add this line
in your shell configuration file (e.g., `~/.bashrc`) to load it automatically or just run the line in 
a terminal window:
```bash
# Execute this in the CTGAN folder, otherwise replace `pwd` with the actual path
export PYTHONPATH=$PYTHONPATH:`pwd`
```

# Other steps to run with CCHIC data
To run within DSH with the CCHIC data as the raw dataset, you first need to:
- Run [this](https://github.com/alan-turing-institute/QUIPP-CC-HIC/blob/develop/quipp-cc-hic/synthesis/synthesis_pipeline.ipynb) 
notebook within DSH. This will generate a dataset with the name `cchic_cleaned.csv` that you then need to place within the `/data` directory in this repo. 
- Place a .json metadata file with the name `cchic_cleaned.json` in the same directory as the dataset. 
This file is not shared here as it might contain sensitive information and needs to be requested from one of the developers of this repo.

# Running the privacy and utility evaluation

### Membership inference
To run a privacy evaluation with respect to the privacy concern of linkability (membership inference)) you can run:
```
python linkage_cli.py -D data/cchic_cleaned -RC tests/linkage/runconfig_cchic.json -O tests/linkage
```
You can edit the configuration file `tests/linkage/runconfig_cchic.json` to choose the 
parameters of the evaluation (more details can be found in the paper):
- `nIter`: Number of independent iterations of the attack
- `sizeRawA`: Size of the shadow raw dataset (sample from the raw data that the intruder has access to)
- `nSynA`: 
- `nShadows`: 
- `sizeRawT`: Size of raw datasets
- `sizeSynT`: Size of synthetic datasets
- `nSynT`: Number of synthetic datasets to generate from each trained generative model
- `nTargets`: Number of targets to randomly pick and perform linkage attack on
- `probIn`: Probability of including each individual to the raw data (assumed known by intruder)
- `Targets`: List of specific targets to perform linkage attack on. Each target is given in the format "IDXXX" where XXX is their sequential order in the raw data.
- `generativeModels`: Dictionary of synthetic methods to test (with parameters for each)
- `sanitisationTechniques`: Dictionary of traditional anonymisation (sanitisation) methods to test (with parameters for each)

The results file produced after successfully running the script is written to `tests/linkage`. 

### Attribute inference
To run a privacy evaluation with respect to the privacy concern of attribute inference you can run:
```
python inference_cli.py -D data/cchic_cleaned -RC tests/inference/runconfig_cchic.json -O tests/inference
```
You can edit the configuration file `tests/inference/runconfig_cchic.json` to choose the 
parameters of the evaluation (more details can be found in the paper):
- `sensitiveAttributes`: Dictionary of the sensitive attributes and whether they should be attacked using classification or regression
- `nIter`: Number of independent iterations of the attack
- `sizeRawT`: Size of raw datasets
- `sizeSynT`: Size of synthetic datasets
- `nSynT`: Number of synthetic datasets to generate from each trained generative model
- `nTargets`: Number of targets to randomly pick and perform inference attack on
- `probIn`: Probability of including each individual to the raw data (assumed known by intruder)
- `positive_label`: A dictionary stating which label is considered the positive label for each of the sensitive attributres
- `prior`: A dictionary stating the prior knowledge of the intruder regarding the values of the sensitive attributes.
- `Targets`: List of specific targets to perform inference attack on. Each target is given in the format "IDXXX" where XXX is their sequential order in the raw data.
- `generativeModels`: Dictionary of synthetic methods to test (with parameters for each)
- `sanitisationTechniques`: Dictionary of traditional anonymisation (sanitisation) methods to test (with parameters for each)

Note that the prior information is set to 0.5 for all cases in the example `tests/inference/cchic_cleaned.json` in this 
repository but the real prior probabilities of the HIV/CIR variables are different. Please request these from the developers.

The results file produced after successfully running the script is written to `tests/inference`

### Utility
To run a utility evaluation with respect to a simple classification task and some univariate utility measures 
(variable means and medians for continuous variables and and frequencies for categorical variables) run:
```
python utility_cli.py -D data/cchic_cleaned -RC tests/utility/runconfig_cchic.json -O tests/utility
```
You can edit the configuration file `tests/utility/runconfig_cchic.json` to choose the 
parameters of the evaluation (more details can be found in the paper):
- `nIter`: Number of independent iterations of the attack
- `sizeRawT`: Size of raw datasets
- `sizeSynT`: Size of synthetic datasets
- `nSynT`: Number of synthetic datasets to generate from each trained generative model
- `nTargets`: Number of targets to randomly pick and evaluate utility on
- `Targets`: List of specific targets to include/not include when doing utility evaluation on. Each target is given in the format "IDXXX" where XXX is their sequential order in the raw data.
- `TestRecords`: List of specific targets to perform utility evaluation on. Each target is given in the format "IDXXX" where XXX is their sequential order in the raw data.
- `generativeModels`: Dictionary of synthetic methods to test (with parameters for each)
- `sanitisationTechniques`: Dictionary of traditional anonymisation (sanitisation) methods to test (with parameters for each)
- `utilityTasks`: Dictionary with types of classification models to train and for which label variables
- `positive_label`: A dictionary stating which label is considered the positive label for each of the attributes targeted as a 
utility task
- `train_fraction`: Percentage of raw data to use as training sample (the rest are withheld as a test sample) 

The results file produced after successfully running the script is written to `tests/utility`.

### Summary and visualisations
In order to generate summary tables from the two privacy evaluation and the utility evaluation and generate 
the respective plots, run the following:
```
python summarise_cli.py -D data/cchic_cleaned -RCI tests/inference/runconfig_cchic -RCL tests/linkage/runconfig_cchic -RCU tests/utility/runconfig_cchic
```
