# Classifying heavy drinkers from resting-state fMRI

Mini-project for UE24CS352A (Machine Learning). It follows the setup of the Stanford
CS229 project "Classifying Adolescent Excessive Alcohol Drinkers from fMRI Data":
predict heavy vs. non-heavy drinking from ICA-parcellated BOLD time series
(25 regions, 269 time steps) plus age, sex and scanner type.

The NCANDA data is access-controlled, so by default the code runs on a simulated
dataset with the same shape and class balance (715 subjects, 122 heavy drinkers).
The simulated group differences are small, set by hand in `fmri/data.py`.
Results on it show how the pipeline behaves, not what is true of real adolescents.

## Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Run

    python main.py                   # about 1.5 minutes on one CPU core
    python main.py --quick           # about 30 seconds, fewer repeats

Tables and figures are written to `results/`. To use real data:

    python main.py --data path/to/folder

where the folder holds `bold.npy` (subjects x timesteps x regions) and `meta.csv`
with columns `label` (0/1), `age`, `sex`, `scanner`, in the same subject order.

## What it does

1. Paper protocol: balance classes by downsampling (244 subjects), 10-fold CV,
   logistic regression fitted with Newton's method, SVMs with four kernels, an MLP.
2. Full-data protocol: all 715 subjects, class-weighted models, hyperparameters
   tuned in an inner loop, 5-fold CV repeated 3 times, ROC-AUC as the main metric.
   Feature sets are signal range, region-to-region correlation, and binned raw series,
   each with and without demographics, and with age/sex/scanner regressed out.
3. A permutation test on the confound-removed connectivity model.

## Layout

    main.py            experiments and command line
    fmri/data.py       simulated data, loader, class balancing
    fmri/features.py   range, connectivity, binned series
    fmri/models.py     Newton logistic regression, confound regression, model factories
    fmri/evaluate.py   cross-validation and permutation test
    fmri/plots.py      figures
