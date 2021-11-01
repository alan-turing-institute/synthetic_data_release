# This script summarises the results from the inference, linkage and utility runs and plots them

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score
from utils.analyse_results import load_results_inference, load_results_linkage, load_results_utility
from argparse import ArgumentParser

plt.style.use('seaborn-whitegrid')
plt.rcParams.update(plt.rcParamsDefault)


def id_conversion(id, dict):
    """Convert id if dictionary is provided, else return original id"""
    if dict is not None:
        return dict["label_dict"][id]
    else:
        return id


def custom_mean(v):
    """Mean that returns nan on value error"""
    try:
        return np.mean(v)
    except ValueError:
        return np.nan


def custom_std(v):
    """SD that returns nan on value error"""
    try:
        return np.std(v)
    except ValueError:
        return np.nan


def errorplots_inference_overall_accuracy(df, sa):
    """Plot inference overall accuracy metrics for attribute sa"""
    data = df[df["SensitiveAttribute"] == sa]
    fig, ax = plt.subplots()
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.1, 0.0) + ax.transData

    acc = plt.errorbar(data["TargetModel"], data["AccRateSynAllMean"],
                       yerr=data["AccRateSynAllSD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans1)
    bacc = plt.errorbar(data["TargetModel"], data["AccBalRateSynAllMean"],
                        yerr=data["AccBalRateSynAllSD"], fmt='o', color='red',
                        ecolor='lightgray', elinewidth=3, capsize=0, transform=trans2)
    plt.title(f"Inference attack on all records ({sa}): Accuracy metrics")
    acc.set_label("Accuracy")
    bacc.set_label("Balanced Accuracy")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/inference_overall_accuracy_{sa}.pdf')


def errorplots_inference_overall_pr(df, sa):
    """Plot inference overall precision and recall for attribute sa"""
    data = df[df["SensitiveAttribute"] == sa]
    fig, ax = plt.subplots()
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.1, 0.0) + ax.transData

    tp = plt.errorbar(data["TargetModel"], data["TPRateSynAllMean"],
                      yerr=data["TPRateSynAllSD"], fmt='o', color='blue',
                      ecolor='lightgray', elinewidth=3, capsize=0, transform=trans1)
    ppv = plt.errorbar(data["TargetModel"], data["PPVRateSynAllMean"],
                       yerr=data["PPVRateSynAllSD"], fmt='o', color='red',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans2)
    plt.title(f"Inference attack on all records ({sa}): Recall and precision metrics")
    tp.set_label("Recall (TPR)")
    ppv.set_label("Precision (PPV)")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/inference_overall_pr_{sa}.pdf')


def errorplots_inference_overall_f1(df, sa):
    """Plot inference overall F1 metrics for attribute sa"""
    data = df[df["SensitiveAttribute"] == sa]
    fig, ax = plt.subplots()
    trans1 = Affine2D().translate(-0.2, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.0, 0.0) + ax.transData
    trans3 = Affine2D().translate(0.2, 0.0) + ax.transData

    f1 = plt.errorbar(data["TargetModel"], data["F1RateSynAllMean"],
                      yerr=data["F1RateSynAllSD"], fmt='o', color='blue',
                      ecolor='lightgray', elinewidth=3, capsize=0, transform=trans1)
    f1macro = plt.errorbar(data["TargetModel"], data["F1MacroRateSynAllMean"],
                           yerr=data["F1MacroRateSynAllSD"], fmt='o', color='red',
                           ecolor='lightgray', elinewidth=3, capsize=0, transform=trans2)
    f1micro = plt.errorbar(data["TargetModel"], data["F1MicroRateSynAllMean"],
                           yerr=data["F1MicroRateSynAllSD"], fmt='o', color='black',
                           ecolor='lightgray', elinewidth=3, capsize=0, transform=trans3)
    plt.title(f"Inference attack on all records ({sa}): F1 metrics")
    f1.set_label("F1 (binary)")
    f1macro.set_label("F1 (macro)")
    f1micro.set_label("F1 (micro)")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/inference_overall_f1_{sa}.pdf')


def errorplots_inference_per_target_acc(df, sa, tid, hash):
    """Plot inference per target accuracy for attribute sa and target tid"""
    data = df[(df["SensitiveAttribute"] == sa) & (df["TargetID"] == tid)]
    fig, ax = plt.subplots()

    acc = plt.errorbar(data["TargetModel"], data["AccSynTotalMean"],
                       yerr=data["AccSynTotalSD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0)
    plt.title(f"Inference attack on target {id_conversion(tid, hash)} ({sa}): Accuracy ")
    acc.set_label("Accuracy")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/inference_per_target_acc_{sa}_{id_conversion(tid, hash)}.pdf')


def errorplots_linkage_per_target_acc(df, tid, hash):
    """Plot linkage per target accuracy for target tid (for all feature set methods)"""
    data_corr = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Correlations")]
    data_hist = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Histogram")]
    data_naive = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Naive")]
    fig, ax = plt.subplots()

    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.0, 0.0) + ax.transData
    trans3 = Affine2D().translate(0.1, 0.0) + ax.transData

    acc_cor = plt.errorbar(data_corr["TargetModel"], data_corr["AccuracySynMean"],
                       yerr=data_corr["AccuracySynSD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans1)
    acc_hist = plt.errorbar(data_hist["TargetModel"], data_hist["AccuracySynMean"],
                       yerr=data_hist["AccuracySynSD"], fmt='o', color='red',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans2)
    acc_naive = plt.errorbar(data_naive["TargetModel"], data_naive["AccuracySynMean"],
                       yerr=data_naive["AccuracySynSD"], fmt='o', color='black',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans3)
    plt.title(f"Membership attack on target {id_conversion(tid, hash)}: Accuracy")
    acc_cor.set_label("Accuracy (Correlation Feature Set)")
    acc_hist.set_label("Accuracy (Histogram Feature Set)")
    acc_naive.set_label("Accuracy (Naive Feature Set)")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/linkage_per_target_acc_{id_conversion(tid, hash)}.pdf')


def errorplots_linkage_per_target_f1(df, tid, hash):
    """Plot linkage per target F1 for target tid (for all feature set methods)"""
    data_corr = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Correlations")]
    data_hist = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Histogram")]
    data_naive = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Naive")]
    fig, ax = plt.subplots()

    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.0, 0.0) + ax.transData
    trans3 = Affine2D().translate(0.1, 0.0) + ax.transData

    f1_cor = plt.errorbar(data_corr["TargetModel"], data_corr["F1RateSynMean"],
                       yerr=data_corr["F1RateSynSD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans1)
    f1_hist = plt.errorbar(data_hist["TargetModel"], data_hist["F1RateSynMean"],
                       yerr=data_hist["F1RateSynSD"], fmt='o', color='red',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans2)
    f1_naive = plt.errorbar(data_naive["TargetModel"], data_naive["F1RateSynMean"],
                       yerr=data_naive["F1RateSynSD"], fmt='o', color='black',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans3)
    plt.title(f"Membership attack on target {id_conversion(tid, hash)}: F1")
    f1_cor.set_label("F1 (Correlation Feature Set)")
    f1_hist.set_label("F1 (Histogram Feature Set)")
    f1_naive.set_label("F1 (Naive Feature Set)")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/linkage_per_target_F1_{id_conversion(tid, hash)}.pdf')


def errorplots_linkage_per_target_recall(df, tid, hash):
    """Plot linkage per target recall for target tid (for all feature set methods)"""
    data_corr = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Correlations")]
    data_hist = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Histogram")]
    data_naive = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Naive")]
    fig, ax = plt.subplots()

    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.0, 0.0) + ax.transData
    trans3 = Affine2D().translate(0.1, 0.0) + ax.transData

    tpr_cor = plt.errorbar(data_corr["TargetModel"], data_corr["TPRateSynMean"],
                       yerr=data_corr["TPRateSynSD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans1)
    tpr_hist = plt.errorbar(data_hist["TargetModel"], data_hist["TPRateSynMean"],
                       yerr=data_hist["TPRateSynSD"], fmt='o', color='red',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans2)
    tpr_naive = plt.errorbar(data_naive["TargetModel"], data_naive["TPRateSynMean"],
                       yerr=data_naive["TPRateSynSD"], fmt='o', color='black',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans3)
    plt.title(f"Membership attack on target {id_conversion(tid, hash)}: True Positive rate (Recall)")
    tpr_cor.set_label("TPR (Correlation Feature Set)")
    tpr_hist.set_label("TPR (Histogram Feature Set)")
    tpr_naive.set_label("TPR (Naive Feature Set)")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/linkage_per_target_TPR_{id_conversion(tid, hash)}.pdf')


def errorplots_linkage_per_target_precision(df, tid, hash):
    """Plot linkage per target precision for target tid (for all feature set methods)"""
    data_corr = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Correlations")]
    data_hist = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Histogram")]
    data_naive = df[(df["TargetID"] == tid) & (df["FeatureSet"] == "Naive")]
    fig, ax = plt.subplots()

    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.0, 0.0) + ax.transData
    trans3 = Affine2D().translate(0.1, 0.0) + ax.transData

    ppv_cor = plt.errorbar(data_corr["TargetModel"], data_corr["PPVRateSynMean"],
                       yerr=data_corr["PPVRateSynSD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans1)
    ppv_hist = plt.errorbar(data_hist["TargetModel"], data_hist["PPVRateSynMean"],
                       yerr=data_hist["PPVRateSynSD"], fmt='o', color='red',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans2)
    ppv_naive = plt.errorbar(data_naive["TargetModel"], data_naive["PPVRateSynMean"],
                       yerr=data_naive["PPVRateSynSD"], fmt='o', color='black',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans3)
    plt.title(f"Memebership attack on target {id_conversion(tid, hash)}: Positive Predictive Value (Precision)")
    ppv_cor.set_label("PPV (Correlation Feature Set)")
    ppv_hist.set_label("PPV (Histogram Feature Set)")
    ppv_naive.set_label("PPV (Naive Feature Set)")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/linkage_per_target_PPV_{id_conversion(tid, hash)}.pdf')


def errorplots_utility_classification(df, label):
    """Plot classification metrics for feature: label"""
    data = df[df["LabelVar"] == label]
    fig, ax = plt.subplots()
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.0, 0.0) + ax.transData
    trans3 = Affine2D().translate(0.1, 0.0) + ax.transData

    acc = plt.errorbar(data["TargetModel"], data["AccuracyMean"],
                       yerr=data["AccuracySD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0, transform=trans1)
    f1 = plt.errorbar(data["TargetModel"], data["F1Mean"],
                        yerr=data["F1SD"], fmt='o', color='red',
                        ecolor='lightgray', elinewidth=3, capsize=0, transform=trans2)
    f1macro = plt.errorbar(data["TargetModel"], data["F1MacroMean"],
                      yerr=data["F1MacroSD"], fmt='o', color='black',
                      ecolor='lightgray', elinewidth=3, capsize=0, transform=trans3)
    plt.title(f"Utility: Predictive model (Label: {label})")
    acc.set_label("Accuracy")
    f1.set_label("F1")
    f1macro.set_label("F1-macro")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/utility_classification_{label}.pdf')


def errorplots_utility_means(df, column):
    """Plot means for variable: column"""
    data = df
    fig, ax = plt.subplots()
    means = plt.errorbar(data["TargetModel"], data[f"{column}Mean"],
                       yerr=data[f"{column}SD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0)
    plt.title(f"Utility: Mean ({column})")
    means.set_label("Mean")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/utility_mean_{column}.pdf')


def errorplots_utility_medians(df, column):
    """Plot medians for variable: column"""
    data = df
    fig, ax = plt.subplots()
    means = plt.errorbar(data["TargetModel"], data[f"{column}Mean"],
                       yerr=data[f"{column}SD"], fmt='o', color='blue',
                       ecolor='lightgray', elinewidth=3, capsize=0)
    plt.title(f"Utility: Median ({column})")
    means.set_label("Median")
    ax.legend(loc='best')
    plt.xlabel("Generative mechanism")
    plt.ylabel("Score")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/utility_median_{column}.pdf')


def errorplots_utility_frequencies(df, column):
    """Plot frequencies for variable: column"""
    methods = df['TargetModel'].unique()
    fig, ax = plt.subplots()
    freqs = []
    trans = []
    ticks = np.linspace(-0.2, 0.2, len(methods))
    from matplotlib.pyplot import cm
    colors = cm.rainbow(np.linspace(0, 1, len(methods)))
    for t in ticks:
        trans.append(Affine2D().translate(t, 0.0) + ax.transData)

    for m, t, c in zip(methods, trans, colors):
        data = df[df["TargetModel"] == m].iloc[0, :]
        try:
            y = data[column + " Mean"]
            x = [f"Cat {n}" for n in np.arange(0, len(y))]
            err = data[column + " SD"]
            freqs.append(plt.errorbar(x, y,
                         yerr=err, fmt='o', color=c,
                         ecolor='lightgray', elinewidth=3, capsize=0, transform=t))
            freqs[-1].set_label(m)
        except TypeError:
            continue
    plt.title(f"Utility: Frequencies ({column})")
    ax.legend(loc='best')
    plt.xlabel("Categories")
    plt.ylabel("Frequency")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'tests/output_plots/utility_freq_{column}.pdf')


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--datapath', '-D', type=str, help='Path a local data file')
    argparser.add_argument('--runconfig_inference', '-RCI', type=str, help='Path to inference runconfig file')
    argparser.add_argument('--runconfig_linkage', '-RCL', type=str, help='Path to linkage runconfig file')
    argparser.add_argument('--runconfig_utility', '-RCU', type=str, help='Path to utility runconfig file')
    argparser.add_argument('--hash', '-H', type=str, help='Path to hash dictionary file', default=None)
    args = argparser.parse_args()

    # Inference attack
    print("Inference attack results:")

    df = load_results_inference("tests/inference", args.datapath, args.runconfig_inference)
    with open(args.runconfig_inference + ".json") as f:
        runconfig = json.load(f)
    pIn = runconfig['probIn']
    with open(args.runconfig_utility + ".json") as f:
        runconfig_utility = json.load(f)
    if args.hash is None:
        hash = None
    else:
        with open(args.hash + ".json") as f:
            hash = json.load(f)

    # Synthetic results
    print("Synthetic dataset (by target)...")
    # Prediction accuracy
    SuccessRateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['AccSynTotal']. \
        agg({'AccSynTotal': {'Mean': np.mean, 'SD': np.std}})
    # True positive rate (Recall)
    TPRateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TPRateSynTotal']. \
        agg({'TPRateSynTotal': {'Mean': np.mean, 'SD': np.std}})
    # Positive Predictive Value (Precision)
    PPVRateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['PPVRateSynTotal']. \
        agg({'PPVRateSynTotal': {'Mean': np.mean, 'SD': np.std}})
    # F1 rate
    F1RateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['F1RateSynTotal']. \
        agg({'F1RateSynTotal': {'Mean': np.mean, 'SD': np.std}})

    # Raw results
    print("Raw dataset (by target)...")
    # Prediction accuracy
    SuccessRateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['AccRawTotal']. \
        agg({'AccRawTotal': {'Mean': np.mean, 'SD': np.std}})
    # True positive rate (Recall)
    TPRateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TPRateRawTotal']. \
        agg({'TPRateRawTotal': {'Mean': np.mean, 'SD': np.std}})
    # Positive Predictive Value (Precision)
    PPVRateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['PPVRateRawTotal']. \
        agg({'PPVRateRawTotal': {'Mean': np.mean, 'SD': np.std}})
    # F1 rate
    F1RateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['F1RateRawTotal']. \
        agg({'F1RateRawTotal': {'Mean': np.mean, 'SD': np.std}})
    # Put everything together
    inference_per_target = SuccessRateSynTotal2 \
        .merge(TPRateSynTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(PPVRateSynTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(F1RateSynTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(SuccessRateRawTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(TPRateRawTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(PPVRateRawTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(F1RateRawTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])

    # Add prior guess metrics
    data_df = pd.read_csv(args.datapath + ".csv")
    for tid in runconfig["Targets"]:
        for sa in runconfig["prior_values"].keys():
            true_values = [str(data_df.loc[int(tid[2:]), sa])] * runconfig["nSynT"]
            labels = list(runconfig["prior_values"][sa].keys())
            values = list(runconfig["prior_values"][sa].values())
            (labels, values) = zip(*runconfig["prior_values"][sa].items())

            prior_guesses = np.random.choice(a=labels, size=[runconfig["nIter"], runconfig["nSynT"]], p=values)
            acc = []
            for i in range(runconfig["nIter"]):
                acc.append(accuracy_score(true_values, prior_guesses[i, :]))

            inference_per_target.loc[(args.datapath.split('/')[1], tid, sa, "Prior Guess")] =  (np.mean(acc), np.std(acc),
                                                                                                np.nan, np.nan,
                                                                                                np.nan, np.nan,
                                                                                                np.nan, np.nan,
                                                                                                np.mean(acc), np.std(acc),
                                                                                                np.nan, np.nan,
                                                                                                np.nan, np.nan,
                                                                                                np.nan, np.nan
                                                                                                )

    # For all rows together (not just targets) - Raw and Synthetic
    print("Raw+Synthetic (for all rows)...")
    F1AllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1RateRawAll']. \
        agg({'F1RateRawAll': {'Mean': np.mean, 'SD': np.std}})
    F1MacroAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1MacroRateRawAll']. \
        agg({'F1MacroRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    F1MicroAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1MicroRateRawAll']. \
        agg({'F1MicroRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    F1AllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1RateSynAll']. \
        agg({'F1RateSynAll': {'Mean': np.mean, 'SD': np.std}})
    F1MacroAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1MacroRateSynAll']. \
        agg({'F1MacroRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    F1MicroAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1MicroRateSynAll']. \
        agg({'F1MicroRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    AccAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccRateRawAll']. \
        agg({'AccRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    AccBalAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccBalRateRawAll']. \
        agg({'AccBalRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    AccAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccRateSynAll']. \
        agg({'AccRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    AccBalAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccBalRateSynAll']. \
        agg({'AccBalRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    TPAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TPRateRawAll']. \
        agg({'TPRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    TPAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TPRateSynAll']. \
        agg({'TPRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    PPVAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['PPVRateRawAll']. \
        agg({'PPVRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    PPVAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['PPVRateSynAll']. \
        agg({'PPVRateSynAll': {'Mean': np.mean, 'SD': np.std}})

    # Put everything together
    inference_overall = F1AllRaw.merge(F1AllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(AccAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(AccAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(TPAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(TPAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(PPVAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(PPVAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(F1MacroAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(F1MacroAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(F1MicroAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(F1MicroAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(AccBalAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel']) \
        .merge(AccBalAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])

    # Add prior guess metrics
    data_df = pd.read_csv(args.datapath + ".csv")
    for sa in runconfig["prior_values"].keys():
        true_values = data_df[sa].astype(str)
        labels = list(runconfig["prior_values"][sa].keys())
        values = list(runconfig["prior_values"][sa].values())
        (labels, values) = zip(*runconfig["prior_values"][sa].items())

        prior_guesses = np.random.choice(a=labels, size=[runconfig["nIter"], true_values.shape[0]], p=values)
        f1, f1macro, f1micro, acc, accbal, tpr, ppv = [], [], [], [], [], [], []
        for i in range(runconfig["nIter"]):
            if len(runconfig["prior_values"][sa].keys()) > 2:
                f1.append(np.nan)
            else:
                f1.append(f1_score(true_values, prior_guesses[i, :],
                                   pos_label=runconfig["positive_label"][sa],
                                   average='binary'))
            f1macro.append(f1_score(true_values, prior_guesses[i, :], average='macro'))
            f1micro.append(f1_score(true_values, prior_guesses[i, :], average='micro'))
            acc.append(accuracy_score(true_values, prior_guesses[i, :]))
            accbal.append(balanced_accuracy_score(true_values, prior_guesses[i, :]))
            tpr.append(recall_score(true_values, prior_guesses[i, :],
                                    labels=[runconfig['positive_label'][sa]],
                                    average='macro'))
            ppv.append(precision_score(true_values, prior_guesses[i, :],
                                       labels=[runconfig['positive_label'][sa]],
                                       average='macro'))

        inference_overall.loc[(args.datapath.split('/')[1], sa, "Prior Guess")] = (np.mean(f1), np.std(f1),
                                                                                   np.mean(f1), np.std(f1),
                                                                                   np.mean(acc), np.std(acc),
                                                                                   np.mean(acc), np.std(acc),
                                                                                   np.mean(tpr), np.std(tpr),
                                                                                   np.mean(tpr), np.std(tpr),
                                                                                   np.mean(ppv), np.std(ppv),
                                                                                   np.mean(ppv), np.std(ppv),
                                                                                   np.mean(f1macro), np.std(f1macro),
                                                                                   np.mean(f1macro), np.std(f1macro),
                                                                                   np.mean(f1micro), np.std(f1micro),
                                                                                   np.mean(f1micro), np.std(f1micro),
                                                                                   np.mean(accbal), np.std(accbal),
                                                                                   np.mean(accbal), np.std(accbal))

    # Linkage attack
    print("\n\nLinkage attack results:")

    df = load_results_linkage("tests/linkage")

    # Synthetic results
    print("Synthetic dataset (by target)...")
    # Prediction accuracy
    SuccessRateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['AccuracySyn']. \
        agg({'AccuracySyn': {'Mean': np.mean, 'SD': np.std}})
    # True positive rate (Recall)
    TPRateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TPRateSyn']. \
        agg({'TPRateSyn': {'Mean': np.mean, 'SD': np.std}})
    # Positive Predictive Value (Precision)
    PPVRateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['PPVRateSyn']. \
        agg({'PPVRateSyn': {'Mean': np.mean, 'SD': np.std}})
    # F1 rate
    F1RateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['F1RateSyn']. \
        agg({'F1RateSyn': {'Mean': np.mean, 'SD': np.std}})

    # Raw results
    print("Raw dataset (by target)...")
    # Prediction accuracy
    SuccessRateRaw2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['AccuracyRaw']. \
        agg({'AccuracyRaw': {'Mean': np.mean, 'SD': np.std}})
    # True positive rate (Recall)
    TPRateRaw2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TPRateRaw']. \
        agg({'TPRateRaw': {'Mean': np.mean, 'SD': np.std}})
    # Positive Predictive Value (Precision)
    PPVRateRaw2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['PPVRateRaw']. \
        agg({'PPVRateRaw': {'Mean': np.mean, 'SD': np.std}})
    # F1 rate
    F1RateRaw2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['F1RateRaw']. \
        agg({'F1RateRaw': {'Mean': np.mean, 'SD': np.std}})

    # Put everything together
    linkage_per_target = SuccessRateSyn2 \
        .merge(TPRateSyn2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(PPVRateSyn2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(F1RateSyn2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(SuccessRateRaw2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(TPRateRaw2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(PPVRateRaw2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(F1RateRaw2, on=['TargetID', 'TargetModel', 'FeatureSet'])

    # Add prior guess metrics
    for tid in runconfig["Targets"]:
        linkage_per_target.loc[(tid, "Prior Guess", "Correlations")] = (
            0.5, 0.0,
            0.5, 0.0,
            0.5, 0.0,
            0.5, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0)
        linkage_per_target.loc[(tid, "Prior Guess", "Histogram")] = (
            0.5, 0.0,
            0.5, 0.0,
            0.5, 0.0,
            0.5, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0)
        linkage_per_target.loc[(tid, "Prior Guess", "Naive")] = (
            0.5, 0.0,
            0.5, 0.0,
            0.5, 0.0,
            0.5, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0,
            1.0, 0.0)

    # For all rows
    print("Raw+Synthetic (for all rows)...")
    # Prediction accuracy
    SuccessRateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['AccuracySyn']. \
        agg({'AccuracySyn': {'Mean': np.mean, 'SD': np.std}})
    # True positive rate (Recall)
    TPRateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['TPRateSyn']. \
        agg({'TPRateSyn': {'Mean': np.mean, 'SD': np.std}})
    # Positive Predictive Value (Precision)
    PPVRateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['PPVRateSyn']. \
        agg({'PPVRateSyn': {'Mean': np.mean, 'SD': np.std}})
    # F1 rate
    F1RateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['F1RateSyn']. \
        agg({'F1RateSyn': {'Mean': np.mean, 'SD': np.std}})
    # Prediction accuracy
    SuccessRateRaw2 = df.groupby(['TargetModel', 'FeatureSet'])['AccuracyRaw']. \
        agg({'AccuracyRaw': {'Mean': np.mean, 'SD': np.std}})
    # True positive rate (Recall)
    TPRateRaw2 = df.groupby(['TargetModel', 'FeatureSet'])['TPRateRaw']. \
        agg({'TPRateRaw': {'Mean': np.mean, 'SD': np.std}})
    # Positive Predictive Value (Precision)
    PPVRateRaw2 = df.groupby(['TargetModel', 'FeatureSet'])['PPVRateRaw']. \
        agg({'PPVRateRaw': {'Mean': np.mean, 'SD': np.std}})
    # F1 rate
    F1RateRaw2 = df.groupby(['TargetModel', 'FeatureSet'])['F1RateRaw']. \
        agg({'F1RateRaw': {'Mean': np.mean, 'SD': np.std}})

    # Put everything together
    linkage_overall = SuccessRateSyn2 \
        .merge(TPRateSyn2, on=['TargetModel', 'FeatureSet']) \
        .merge(PPVRateSyn2, on=['TargetModel', 'FeatureSet']) \
        .merge(F1RateSyn2, on=['TargetModel', 'FeatureSet']) \
        .merge(SuccessRateRaw2, on=['TargetModel', 'FeatureSet']) \
        .merge(TPRateRaw2, on=['TargetModel', 'FeatureSet']) \
        .merge(PPVRateRaw2, on=['TargetModel', 'FeatureSet']) \
        .merge(F1RateRaw2, on=['TargetModel', 'FeatureSet'])

    # Utility
    print("\n\nUtility results:")

    df = load_results_utility("tests/utility")[1]

    # All methods (Synthetic, raw, sanitised)
    print("Synthetic+Raw for Mean/Median/Frequencies...")
    # calculate means and SD for mean/median/frequencies
    Means = {var[5:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].
        agg({var[5:]: {'Mean': np.mean, 'SD': np.std}})
             for var in [c for c in df.columns if c.startswith('Mean_')]}
    Medians = {var[7:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].
        agg({var[7:]: {'Mean': np.mean, 'SD': np.std}})
               for var in [c for c in df.columns if c.startswith('Median_')]}
    for c in df.columns:
        if c.startswith('Freq_'):
            df[c] = df[c].apply(np.array)
    Frequencies_Means = {
        var[5:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].apply(np.array)
            .rename(columns={var: var[5:] + " Means"})
        for var in [c for c in df.columns if c.startswith('Freq_')]}
    Frequencies_SDs = {
        var[5:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].apply(np.array)
            .rename(columns={var: var[5:] + " SD"})
        for var in [c for c in df.columns if c.startswith('Freq_')]}

    # get categorical columns
    columns_categorical = list(Frequencies_Means.keys())

    for k in Frequencies_Means.keys():
        try:
            Frequencies_Means[k] = pd.DataFrame(Frequencies_Means[k].apply(custom_mean))
            Frequencies_Means[k].rename(columns={0: k + " Mean"}, inplace=True)
        except ValueError:
            Frequencies_Means[k] = np.nan
            Frequencies_Means[k].rename(columns={0: k + " Mean"}, inplace=True)
    for k in Frequencies_SDs.keys():
        try:
            Frequencies_SDs[k] = pd.DataFrame(Frequencies_SDs[k].apply(custom_std))
            Frequencies_SDs[k].rename(columns={0: k + " SD"}, inplace=True)
        except ValueError:
            Frequencies_SDs[k] = np.nan
            Frequencies_SDs[k].rename(columns={0: k + " SD"}, inplace=True)

    print("Synthetic+Raw for Classification accuracy...")
    # get means and SDs
    Accuracy = df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])['Accuracy']. \
        agg({'Accuracy': {'Mean': np.mean, 'SD': np.std}})
    F1 = df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])['F1']. \
        agg({'F1': {'Mean': np.mean, 'SD': np.std}})
    F1Macro = df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])['F1Macro']. \
        agg({'F1Macro': {'Mean': np.mean, 'SD': np.std}})

    # Put everything together
    utility_classification_overall = Accuracy.merge(F1, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])\
        .merge(F1Macro, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])

    # merge all dataframes
    i = 0
    for k, v in Means.items():
        i = i + 1
        if i > 1:
            temp = temp.merge(v, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])
        else:
            temp = v
    utility_means_overall = temp

    # get numerical columns
    columns_numerical = set([c[0] for c in utility_means_overall.columns])

    # merge all dataframes
    i = 0
    for k, v in Medians.items():
        i = i + 1
        if i > 1:
            temp = temp.merge(v, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])
        else:
            temp = v

    utility_medians_overall = temp

    # merge all dataframes
    i = 0
    for k, v in Frequencies_Means.items():
        i = i + 1
        if i > 1:
            temp = temp.merge(v, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])
        else:
            temp = v

    for k, v in Frequencies_SDs.items():
        temp = temp.merge(v, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])

    utility_freq_overall = temp

    # Writing results to disk
    print("Writing results to disk...")
    all_metrics = {
        "inference_per_target": inference_per_target,
        "inference_overall": inference_overall,
        "linkage_per_target": linkage_per_target,
        "linkage_overall": linkage_overall,
        "utility_classification_overall": utility_classification_overall,
        "utility_means_overall": utility_means_overall,
        "utility_medians_overall": utility_medians_overall,
        "utility_freq_overall": utility_freq_overall}

    inference_per_target.to_csv("tests/output_dataframes/inference_per_target.csv")
    inference_overall.to_csv("tests/output_dataframes/inference_overall.csv")
    linkage_per_target.to_csv("tests/output_dataframes/linkage_per_target.csv")
    linkage_overall.to_csv("tests/output_dataframes/linkage_overall.csv")
    utility_classification_overall.to_csv("tests/output_dataframes/utility_classification_overall.csv")
    utility_means_overall.to_csv("tests/output_dataframes/utility_means_overall.csv")
    utility_medians_overall.to_csv("tests/output_dataframes/utility_medians_overall.csv")
    utility_freq_overall.to_csv("tests/output_dataframes/utility_freq_overall.csv")

    # Plot results
    print("Generating plots...")
    # convert index to columns and rename columns
    inference_overall = inference_overall.reset_index()
    inference_overall.columns = [''.join(col) for col in inference_overall.columns]
    inference_per_target = inference_per_target.reset_index()
    inference_per_target.columns = [''.join(col) for col in inference_per_target.columns]
    linkage_per_target = linkage_per_target.reset_index()
    linkage_per_target.columns = [''.join(col) for col in linkage_per_target.columns]
    utility_classification_overall = utility_classification_overall.reset_index()
    utility_classification_overall.columns = [''.join(col) for col in utility_classification_overall.columns]
    utility_means_overall = utility_means_overall.reset_index()
    utility_means_overall.columns = [''.join(col) for col in utility_means_overall.columns]
    utility_medians_overall = utility_medians_overall.reset_index()
    utility_medians_overall.columns = [''.join(col) for col in utility_medians_overall.columns]
    utility_freq_overall = utility_freq_overall.reset_index()

    # Plot inference performance plots
    for sa in inference_overall["SensitiveAttribute"]:
        errorplots_inference_overall_accuracy(inference_overall, sa)
        errorplots_inference_overall_pr(inference_overall, sa)
        errorplots_inference_overall_f1(inference_overall, sa)
        for tid in runconfig["Targets"]:
            errorplots_inference_per_target_acc(inference_per_target, sa, tid, hash)

    # Plot linkage performance plots
    for tid in runconfig["Targets"]:
        errorplots_linkage_per_target_acc(linkage_per_target, tid, hash)
        errorplots_linkage_per_target_f1(linkage_per_target, tid, hash)
        errorplots_linkage_per_target_recall(linkage_per_target, tid, hash)
        errorplots_linkage_per_target_precision(linkage_per_target, tid, hash)

    # Plot utility (classification) plots
    for label in runconfig_utility["utilityTasks"]["RandForestClass"]:
        errorplots_utility_classification(utility_classification_overall, label[0])

    # Plot utility (means/medians) plots
    for column in columns_numerical:
        errorplots_utility_means(utility_means_overall, column)
        errorplots_utility_medians(utility_medians_overall, column)

    # Plot utility (frequencies) plots
    for column in columns_categorical:
        errorplots_utility_frequencies(utility_freq_overall, column)


if __name__ == "__main__":
    main()
