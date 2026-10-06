import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from fmri import features, models, plots
from fmri.data import balanced_subset, load_dataset, make_synthetic
from fmri.evaluate import (permutation_test, run_cv, run_paper_cv, summarize)


def paper_protocol(bold, meta, seed):
    """
    Run the paper-based classification protocol.

    The protocol balances the dataset and evaluates the models
    using 10-fold cross-validation.

    Parameters
    ----------
    bold : np.ndarray
        Resting-state fMRI BOLD time-series data.
    meta : pd.DataFrame
        Subject metadata containing labels and demographic information.
    seed : int
        Random seed used for reproducible evaluation.

    Returns
    -------
     pd.DataFrame
        Classification performance metrics for each model.
    """
    idx = balanced_subset(meta, seed)
    b, m = bold[idx], meta.iloc[idx].reset_index(drop=True)
    y = m["label"].to_numpy()
    R, S, D = features.range_features(b), features.binned_series(b), features.demographics(m)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    newton = lambda: make_pipeline(StandardScaler(), models.NewtonLogisticRegression())
    specs = [
        ("Logistic regression (range + demographics)", np.c_[R, D], newton()),
        ("Logistic regression (range + sex, scanner)", np.c_[R, D[:, 1:]], newton()),
        ("SVM linear", np.c_[R, D], models.svm_kernel("linear")),
        ("SVM poly (deg 2)", np.c_[R, D], models.svm_kernel("poly")),
        ("SVM sigmoid", np.c_[R, D], models.svm_kernel("sigmoid")),
        ("SVM rbf", np.c_[R, D], models.svm_kernel("rbf")),
        ("MLP on binned time series", S, models.mlp()),
    ]
    rows = []
    for name, X, model in specs:
        s = summarize(run_paper_cv(model, X, y, seed))
        rows.append({"model": name, "n": len(y),
                     "train_acc": s["train_accuracy"][0], "test_acc": s["test_accuracy"][0],
                     "test_acc_sd": s["test_accuracy"][1], "test_f1": s["test_f1"][0]})
        print(f"  {name:46s} train {rows[-1]['train_acc']:.3f}  test {rows[-1]['test_acc']:.3f}")
    return pd.DataFrame(rows)


def full_protocol(bold, meta, seed, repeats):
    """All subjects, class-weighted models, nested tuning, repeated CV."""
    y = meta["label"].to_numpy()
    R = features.range_features(bold)
    C = features.connectivity_features(bold)
    S = features.binned_series(bold)
    D = features.demographics(meta)

    lr, lr_res = models.logreg(), models.logreg(residualize=True)
    g = models.LOGREG_GRID
    specs = [
        ("Age only", "demographics only", D[:, :1], lr, g),
        ("Age, sex, scanner", "demographics only", D, lr, g),
        ("Range", "fMRI only", R, lr, g),
        ("Connectivity", "fMRI only", C, lr, g),
        ("Connectivity, RBF-SVM", "fMRI only", C, models.rbf_svm(), models.SVM_GRID),
        ("MLP on time series", "fMRI only", S, models.mlp(), None),
        ("Range + demographics", "with demographics", np.c_[R, D], lr, g),
        ("Connectivity + demographics", "with demographics", np.c_[C, D], lr, g),
        ("Range, confounds removed", "confounds removed", np.c_[R, D], lr_res, g),
        ("Connectivity, confounds removed", "confounds removed", np.c_[C, D], lr_res, g),
    ]
    rows = []
    for name, group, X, model, grid in specs:
        s = summarize(run_cv(model, X, y, grid, n_repeats=repeats, seed=seed))
        rows.append({"model": name, "group": group,
                     "auc": s["auc"][0], "auc_sd": s["auc"][1],
                     "bal_acc": s["bal_acc"][0], "bal_acc_sd": s["bal_acc"][1],
                     "f1": s["f1"][0]})
        print(f"  {name:34s} AUC {s['auc'][0]:.3f}  bal.acc {s['bal_acc'][0]:.3f}")
    return pd.DataFrame(rows), np.c_[C, D]


def main():
    """ 
    Run the fMRI alcohol classification experiments.

    Parses command-line arguments, loads the dataset, runs the
    classification protocols, performs the permutation test, and
    writes tables and figures to the output directory.
    """
    ap = argparse.ArgumentParser(description="Classify heavy drinkers from resting-state fMRI features")
    ap.add_argument("--data", help="folder with bold.npy and meta.csv (default: simulated data)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="1 CV repeat, 20 permutations")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    bold, meta = load_dataset(args.data) if args.data else make_synthetic(seed=args.seed)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    print(f"{len(meta)} subjects, {int(meta['label'].sum())} heavy drinkers, "
          f"series shape {bold.shape[1]} x {bold.shape[2]}")

    plots.timeseries(bold, meta, out)
    plots.age_distribution(meta, out)

    print("\nPaper protocol (balanced subset, 10-fold CV)")
    paper = paper_protocol(bold, meta, args.seed)
    paper.to_csv(out / "paper_protocol.csv", index=False)

    print("\nFull-data protocol (nested tuning, repeated stratified CV)")
    full, X_conn = full_protocol(bold, meta, args.seed, repeats=1 if args.quick else 3)
    full.to_csv(out / "full_protocol.csv", index=False)
    plots.model_comparison(list(full[["model", "group", "auc", "auc_sd"]].itertuples(index=False)), out)

    n_perm = 20 if args.quick else 100
    print(f"\nPermutation test, connectivity with confounds removed ({n_perm} shuffles)")
    score, perms, p = permutation_test(models.logreg(residualize=True), X_conn,
                                       meta["label"].to_numpy(), n_perm, args.seed)
    print(f"  AUC {score:.3f}, p = {p:.3f}")
    plots.permutation(score, perms, p, out)
    pd.DataFrame({"auc": [score], "p_value": [p], "n_permutations": [n_perm]}).to_csv(
        out / "permutation_test.csv", index=False)

    print(f"\nTables and figures written to {out}/")


if __name__ == "__main__":
    main()
