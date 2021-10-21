import json
import numpy as np
import pandas as pd
from utils.analyse_results import load_results_inference, load_results_linkage, load_results_utility
from argparse import ArgumentParser


def custom_mean(v):
    try:
        return np.mean(v)
    except ValueError:
        return np.nan


def custom_std(v):
    try:
        return np.std(v)
    except ValueError:
        return np.nan


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--datapath', '-D', type=str, help='Path a local data file')
    argparser.add_argument('--runconfig_inference', '-RCI', type=str, help='Path to inference runconfig file')
    argparser.add_argument('--runconfig_linkage', '-RCL', type=str, help='Path to linkage runconfig file')
    args = argparser.parse_args()

    # Inference attack
    print("Inference attack results:")

    df = load_results_inference("tests/inference", args.datapath, args.runconfig_inference)
    with open(args.runconfig_inference + ".json") as f:
        runconfig = json.load(f)
    pIn = runconfig['probIn']

    # Synthetic results
    print("Synthetic dataset (by target)...")
    # Prediction accuracy
    SuccessRateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['AccSynTotal'].\
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

    # For all rows together (not just targets) - Raw and Synthetic
    print("Raw+Synthetic (for all rows)...")
    F1AllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1RateRawAll']. \
        agg({'F1RateRawAll': {'Mean': np.mean, 'SD': np.std}})
    F1AllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1RateSynAll']. \
        agg({'F1RateSynAll': {'Mean': np.mean, 'SD': np.std}})
    AccAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccRateRawAll']. \
        agg({'AccRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    AccAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccRateSynAll']. \
        agg({'AccRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    TPAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TPRateRawAll']. \
        agg({'TPRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    TPAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TPRateSynAll']. \
        agg({'TPRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    # Put everything together
    inference_overall = F1AllRaw.merge(F1AllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])\
        .merge(AccAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])\
        .merge(AccAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])\
        .merge(TPAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])\
        .merge(TPAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])

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
    Accuracy = df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])['Accuracy']. \
               agg({'Accuracy': {'Mean': np.mean, 'SD': np.std}})
    F1 = df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])['F1']. \
        agg({'F1': {'Mean': np.mean, 'SD': np.std}})

    # Put everything together
    utility_classification_overall = Accuracy.merge(F1, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])

    i = 0
    for k, v in Means.items():
        i = i + 1
        if i > 1:
            temp = temp.merge(v, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])
        else:
            temp = v
    utility_means_overall = temp

    i = 0
    for k, v in Medians.items():
        i = i + 1
        if i > 1:
            temp = temp.merge(v, on=['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])
        else:
            temp = v

    utility_medians_overall = temp

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


if __name__ == "__main__":
    main()
