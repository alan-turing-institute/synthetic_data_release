# Privacy evaluation framework for synthetic data publishing
A practical framework to evaluate the privacy-utility tradeoff of synthetic data publishing 

Based on "Synthetic Data - Anonymisation Groundhog Day, Theresa Stadler, Bristena Oprisanu, and Carmela Troncoso, [arXiv](https://arxiv.org/abs/2011.07018), 2020"

This fork contains changes to the code that allow it to run within the UCL Data Safe Haven and used for assessing the privacy and utility of the CCHIC database, see [this](https://github.com/alan-turing-institute/QUIPP-CC-HIC) repository for more details on the project.

# Attack models
The module `attack_models` so far includes

A privacy adversary to test for privacy gain with respect to linkage attacks modelled as a membership inference attack `MIAAttackClassifier`.

A simple attribute inference attack `AttributeInferenceAttack` that aims to infer a target's sensitive value given partial knowledge about the target record

# Generative models
The module `generative_models` so far includes:   
- `IndependentHistogramModel`: An independent histogram model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `BayesianNetModel`: A generative model based on a Bayesian Network adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `GaussianMixtureModel`: A simple Gaussian Mixture model taken from the [sklearn library](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- `CTGAN`: A conditional tabular generative adversarial network that integrates the CTGAN model from [CTGAN](https://github.com/sdv-dev/CTGAN)  
- `PATE-GAN`: A differentially private generative adversarial network adapted from its original [implementation](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/)

# Setup
## Requirements
The framework and its building blocks have been developed and tested under Python 3.6 and 3.7

We recommend to create a virtual environment for installing all dependencies and running the code
```
python3 -m venv pyvenv3
source pyvenv3/bin/activate
pip install -r requirements.txt
```

## Dependencies
The `CTGAN` model depends on a fork of the original model training algorithm that can be found here
[CTGAN-Turing](https://github.com/alan-turing-institute/CTGAN)

To install the correct version clone the repository above and run
```
cd CTGAN
make install
```

You then need to add the path to this directory to your python path by typing the following line in a command window (note this will only work for the current command window session so you need to do it every time you open a new command window):
```
set PYTHONPATH=%PYTHONPATH%;ctgan_path
```
where `ctgan_path` is the path to this directory (e.g. `N:\projects\CTGAN`). Note this applies to the DSH Windows environment.

For Linux/Mac environments, you can also add this line
in your shell configuration file (e.g., `~/.bashrc`) to load it automatically.
```bash
# Execute this in the CTGAN folder, otherwise replace `pwd` with the actual path
export PYTHONPATH=$PYTHONPATH:`pwd`
```

To test your installation try to run
```
import ctgan
```
from within your virtualenv `python`

# Example runs
To run within CCHIC DSH with the CCHIC data as raw dataset, you need to first run [this](https://github.com/alan-turing-institute/QUIPP-CC-HIC/blob/develop/quipp-cc-hic/synthesis/synthesis_pipeline.ipynb) notebook within DSH. 
This will generate a dataset that you then need to place within the `/data` directory in this repo. You also need to place a .json metadata file with the same name as the dataset in the same directory. 
This is not shared here as it might contain sensitive information and needs to be requested from one of the developers of this repo.

To run a privacy evaluation with respect to the privacy concern of linkability you can run

```
python linkage_cli.py -D data/cchic_cleaned -RC tests/linkage/runconfig_cchic.json -O tests/linkage
```

The results file produced after successfully running the script will be written to `tests/linkage` and can be parsed with the function `load_results_linkage` provided in `utils/analyse_results.py`. 


To run a privacy evaluation with respect to the privacy concern of inference you can run

```
python inference_cli.py -D data/cchic_cleaned -RC tests/inference/runconfig_cchic.json -O tests/inference
```

Note that the prior information is set to 0.5 for all cases here but the real prior probabilities of HIC/CIR disease are different. 
Please request these from the developers. 
The results file produced after successfully running the script can be parsed with the function `load_results_inference` provided in `utils/analyse_results.py`.


To run a utility evaluation with respect to a simple classification task as utility function run

```
python utility_cli.py -D data/cchic_cleaned -RC tests/utility/runconfig_cchic.json -O tests/utility
```

The results file produced after successfully running the script can be parsed with the function `load_results_utility` provided in `utils/analyse_results.py`.

In order to print a summary of all the attacks you ran for the CCHIC dataset, run the following:

```
python summarise_cli.py -D data/cchic_cleaned -RCI tests/inference/runconfig_cchic -RCL tests/linkage/runconfig_cchic
```
