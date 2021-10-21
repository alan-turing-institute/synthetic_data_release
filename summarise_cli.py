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
    print("Inference attack results")

    df = load_results_inference("tests/inference", args.datapath, args.runconfig_inference)
    with open(args.runconfig_inference + ".json") as f:
        runconfig = json.load(f)
    pIn = runconfig['probIn']

    # Synthetic results
    print("Synthetic dataset")

    # By target
    print("By target:")

    # Prediction accuracy
    # 1. Summing all iterations
    MatchesSynIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum() + \
                   df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesSynIn'].sum()
    TotalSynIn = MatchesSynIn + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynIn'].sum() + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynIn'].sum()
    SuccessRateSynIn = MatchesSynIn / TotalSynIn
    MatchesSynOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum() + \
                   df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesSynOut'].sum()
    TotalSynOut = MatchesSynOut + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynOut'].sum() + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynOut'].sum()
    SuccessRateSynOut = MatchesSynOut / TotalSynOut
    SuccessRateSynTotal = SuccessRateSynIn * pIn + SuccessRateSynOut * (1 - pIn)
    print(f'Prediction accuracy (1)\n{SuccessRateSynTotal}')
    # 2. Mean/STD from different iterations
    SuccessRateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['AccSynTotal'].\
        agg({'AccSynTotal': {'Mean': np.mean, 'SD': np.std}})
    print(f'Prediction accuracy (2)\n{SuccessRateSynTotal2}')

    # True positive rate (Recall)
    # 1. Summing all iterations
    TruePositivesSynIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum()
    PositiveLabelsSynIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum() + \
                          df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynIn'].sum()
    TPRateSynIn = TruePositivesSynIn / PositiveLabelsSynIn
    TruePositivesSynOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum()
    PositiveLabelsSynOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum() + \
                          df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynOut'].sum()
    TPRateSynOut = TruePositivesSynOut / PositiveLabelsSynOut
    TPRateSynTotal = TPRateSynIn * pIn + TPRateSynOut * (1 - pIn)
    print(f'True positive rate (Recall) (1)\n{TPRateSynTotal}')
    # 2. Mean/STD from different iterations
    TPRateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TPRateSynTotal']. \
        agg({'TPRateSynTotal': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) (2)\n{TPRateSynTotal2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations
    PositivesSynIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum() + \
                     df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynIn'].sum()
    PPVRateSynIn = TruePositivesSynIn / PositivesSynIn
    PositivesSynOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum() + \
                     df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynOut'].sum()
    PPVRateSynOut = TruePositivesSynOut / PositivesSynOut
    PPVRateSynTotal = PPVRateSynIn * pIn + PPVRateSynOut * (1 - pIn)
    print(f'Positive Predictive Value (Precision) (1)\n{PPVRateSynTotal}')
    # 2. Mean/STD from different iterations
    PPVRateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['PPVRateSynTotal']. \
        agg({'PPVRateSynTotal': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) (2)\n{PPVRateSynTotal2}')

    # F1 rate
    # 1. Summing all iterations
    F1RateSynIn = 2 * (PPVRateSynIn * TPRateSynIn) / (PPVRateSynIn + TPRateSynIn)
    F1RateSynOut = 2 * (PPVRateSynOut * TPRateSynOut) / (PPVRateSynOut + TPRateSynOut)
    F1RateSynTotal = F1RateSynIn * pIn + F1RateSynOut * (1 - pIn)
    print(f'F1 rate (1)\n{F1RateSynTotal}')
    # 2. Mean/STD from different iterations
    F1RateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['F1RateSynTotal']. \
        agg({'F1RateSynTotal': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate (2)\n{F1RateSynTotal2}')


    # Raw results
    print("Raw dataset")

    # By target
    print("By target:")

    # Prediction accuracy
    # 1. Summing all iterations
    MatchesRawIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                       'TruePositivesRawIn'].sum() + \
                   df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesRawIn'].sum()
    TotalRawIn = MatchesRawIn + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawIn'].sum() + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawIn'].sum()
    SuccessRateRawIn = MatchesRawIn / TotalRawIn
    MatchesRawOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                        'TruePositivesRawOut'].sum() + \
                    df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                        'TrueNegativesRawOut'].sum()
    TotalRawOut = MatchesRawOut + \
                  df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                      'FalsePositivesRawOut'].sum() + \
                  df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawOut'].sum()
    SuccessRateRawOut = MatchesRawOut / TotalRawOut
    SuccessRateRawTotal = SuccessRateRawIn * pIn + SuccessRateRawOut * (1 - pIn)
    print(f'Prediction accuracy (1)\n{SuccessRateRawTotal}')
    # 2. Mean/STD from different iterations
    SuccessRateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['AccRawTotal']. \
        agg({'AccRawTotal': {'Mean': np.mean, 'SD': np.std}})
    print(f'Prediction accuracy (2)\n{SuccessRateRawTotal2}')

    # True positive rate (Recall)
    # 1. Summing all iterations
    TruePositivesRawIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
        'TruePositivesRawIn'].sum()
    PositiveLabelsRawIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                              'TruePositivesRawIn'].sum() + \
                          df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                              'FalseNegativesRawIn'].sum()
    TPRateRawIn = TruePositivesRawIn / PositiveLabelsRawIn
    TruePositivesRawOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
        'TruePositivesRawOut'].sum()
    PositiveLabelsRawOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                               'TruePositivesRawOut'].sum() + \
                           df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                               'FalseNegativesRawOut'].sum()
    TPRateRawOut = TruePositivesRawOut / PositiveLabelsRawOut
    TPRateRawTotal = TPRateRawIn * pIn + TPRateRawOut * (1 - pIn)
    print(f'True positive rate (Recall) (1)\n{TPRateRawTotal}')
    # 2. Mean/STD from different iterations
    TPRateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TPRateRawTotal']. \
        agg({'TPRateRawTotal': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) (2)\n{TPRateRawTotal2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations
    PositivesRawIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                         'TruePositivesRawIn'].sum() + \
                     df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                         'FalsePositivesRawIn'].sum()
    PPVRateRawIn = TruePositivesRawIn / PositivesRawIn
    PositivesRawOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                          'TruePositivesRawOut'].sum() + \
                      df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])[
                          'FalsePositivesRawOut'].sum()
    PPVRateRawOut = TruePositivesRawOut / PositivesRawOut
    PPVRateRawTotal = PPVRateRawIn * pIn + PPVRateRawOut * (1 - pIn)
    print(f'Positive Predictive Value (Precision) (1)\n{PPVRateRawTotal}')
    # 2. Mean/STD from different iterations
    PPVRateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['PPVRateRawTotal']. \
        agg({'PPVRateRawTotal': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) (2)\n{PPVRateRawTotal2}')

    # F1 rate
    # 1. Summing all iterations
    F1RateRawIn = 2 * (PPVRateRawIn * TPRateRawIn) / (PPVRateRawIn + TPRateRawIn)
    F1RateRawOut = 2 * (PPVRateRawOut * TPRateRawOut) / (PPVRateRawOut + TPRateRawOut)
    F1RateRawTotal = F1RateRawIn * pIn + F1RateRawOut * (1 - pIn)
    print(f'F1 rate (1)\n{F1RateRawTotal}')
    # 2. Mean/STD from different iterations
    F1RateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['F1RateRawTotal']. \
        agg({'F1RateRawTotal': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate (2)\n{F1RateRawTotal2}')

    # Put everything together

    inference_per_target = SuccessRateSynTotal2 \
        .merge(TPRateSynTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(PPVRateSynTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(F1RateSynTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(SuccessRateRawTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(TPRateRawTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(PPVRateRawTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel']) \
        .merge(F1RateRawTotal2, on=['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])




    # Rates for all rows
    F1AllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1RateRawAll']. \
        agg({'F1RateRawAll': {'Mean': np.mean, 'SD': np.std}})
    print(f"Raw F1 All:\n{F1AllRaw}")

    F1AllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1RateSynAll']. \
        agg({'F1RateSynAll': {'Mean': np.mean, 'SD': np.std}})
    print(f"Synthetic F1 All:\n{F1AllSyn}")
    AccAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccRateRawAll']. \
        agg({'AccRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    print(f"Raw Accuracy All:\n{AccAllRaw}")

    AccAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccRateSynAll']. \
        agg({'AccRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    print(f"Synthetic Accuracy All:\n{AccAllSyn}")
    TPAllRaw = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TPRateRawAll']. \
        agg({'TPRateRawAll': {'Mean': np.mean, 'SD': np.std}})
    print(f"Raw TP Rate All:\n{TPAllRaw}")

    TPAllSyn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TPRateSynAll']. \
        agg({'TPRateSynAll': {'Mean': np.mean, 'SD': np.std}})
    print(f"Synthetic TP Rate All:\n{TPAllSyn}")

    # Put everything together

    inference_overall = F1AllRaw.merge(F1AllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])\
        .merge(AccAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])\
        .merge(AccAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])\
        .merge(TPAllRaw, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])\
        .merge(TPAllSyn, on=['Dataset', 'SensitiveAttribute', 'TargetModel'])















    # Linkage attack
    print("\n\n\n\n\nLinkage attack results")

    df = load_results_linkage("tests/linkage")

    # Synthetic results
    print("Synthetic dataset")

    # By target
    print("By target:")

    # Prediction accuracy
    # 1. Summing all iterations
    MatchesSyn = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TruePositivesSyn'].sum() + \
                   df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TrueNegativesSyn'].sum()
    TotalSyn = MatchesSyn + \
                 df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['FalsePositivesSyn'].sum() + \
                 df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['FalseNegativesSyn'].sum()
    SuccessRateSyn = MatchesSyn / TotalSyn
    print(f'Prediction accuracy (1)\n{SuccessRateSyn}')
    # 2. Mean/STD from different iterations
    SuccessRateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['AccuracySyn']. \
        agg({'AccuracySyn': {'Mean': np.mean, 'SD': np.std}})
    print(f'Prediction accuracy (2)\n{SuccessRateSyn2}')

    # True positive rate (Recall)
    # 1. Summing all iterations
    TruePositivesSyn = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TruePositivesSyn'].sum()
    PositiveLabelsSyn = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TruePositivesSyn'].sum() + \
                          df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])[
                              'FalseNegativesSyn'].sum()
    TPRateSyn = TruePositivesSyn / PositiveLabelsSyn
    print(f'True positive rate (Recall) (1)\n{TPRateSyn}')
    # 2. Mean/STD from different iterations
    TPRateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TPRateSyn']. \
        agg({'TPRateSyn': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) (2)\n{TPRateSyn2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations
    PositivesSyn = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])[
                         'TruePositivesSyn'].sum() + \
                     df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])[
                         'FalsePositivesSyn'].sum()
    PPVRateSyn = TruePositivesSyn / PositivesSyn
    print(f'Positive Predictive Value (Precision) (1)\n{PPVRateSyn}')
    # 2. Mean/STD from different iterations
    PPVRateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['PPVRateSyn']. \
        agg({'PPVRateSyn': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) (2)\n{PPVRateSyn2}')

    # F1 rate
    # 1. Summing all iterations
    F1RateSyn = 2 * (PPVRateSyn * TPRateSyn) / (PPVRateSyn + TPRateSyn)
    print(f'F1 rate (1)\n{F1RateSyn}')
    # 2. Mean/STD from different iterations
    F1RateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['F1RateSyn']. \
        agg({'F1RateSyn': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate (2)\n{F1RateSyn2}')


    # Raw

    # Prediction accuracy
    # 1. Summing all iterations
    MatchesRaw = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TruePositivesRaw'].sum() + \
                 df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TrueNegativesRaw'].sum()
    TotalRaw = MatchesRaw + \
               df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['FalsePositivesRaw'].sum() + \
               df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['FalseNegativesRaw'].sum()
    SuccessRateRaw = MatchesRaw / TotalRaw
    print(f'Prediction accuracy (1)\n{SuccessRateRaw}')
    # 2. Mean/STD from different iterations
    SuccessRateRaw2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['AccuracyRaw']. \
        agg({'AccuracyRaw': {'Mean': np.mean, 'SD': np.std}})
    print(f'Prediction accuracy (2)\n{SuccessRateRaw2}')

    # True positive rate (Recall)
    # 1. Summing all iterations
    TruePositivesRaw = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TruePositivesRaw'].sum()
    PositiveLabelsRaw = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TruePositivesRaw'].sum() + \
                        df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])[
                            'FalseNegativesRaw'].sum()
    TPRateRaw = TruePositivesRaw / PositiveLabelsRaw
    print(f'True positive rate (Recall) (1)\n{TPRateRaw}')
    # 2. Mean/STD from different iterations
    TPRateRaw2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['TPRateRaw']. \
        agg({'TPRateRaw': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) (2)\n{TPRateRaw2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations
    PositivesRaw = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])[
                       'TruePositivesRaw'].sum() + \
                   df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])[
                       'FalsePositivesRaw'].sum()
    PPVRateRaw = TruePositivesRaw / PositivesRaw
    print(f'Positive Predictive Value (Precision) (1)\n{PPVRateRaw}')
    # 2. Mean/STD from different iterations
    PPVRateRaw2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['PPVRateRaw']. \
        agg({'PPVRateRaw': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) (2)\n{PPVRateRaw2}')

    # F1 rate
    # 1. Summing all iterations
    F1RateRaw = 2 * (PPVRateRaw * TPRateRaw) / (PPVRateRaw + TPRateRaw)
    print(f'F1 rate (1)\n{F1RateRaw}')
    # 2. Mean/STD from different iterations
    F1RateRaw2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['F1RateRaw']. \
        agg({'F1RateRaw': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate (2)\n{F1RateRaw2}')


    # Put everything together

    linkage_per_target = SuccessRateSyn2 \
        .merge(TPRateSyn2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(PPVRateSyn2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(F1RateSyn2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(SuccessRateRaw2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(TPRateRaw2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(PPVRateRaw2, on=['TargetID', 'TargetModel', 'FeatureSet']) \
        .merge(F1RateRaw2, on=['TargetID', 'TargetModel', 'FeatureSet'])



    # For all targets combined
    print("For all targets combined:")

    # Prediction accuracy
    # 1. Summing all iterations and targets
    MatchesSyn = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesSyn'].sum() + \
                 df.groupby(['TargetModel', 'FeatureSet'])['TrueNegativesSyn'].sum()
    TotalSyn = MatchesSyn + \
                 df.groupby(['TargetModel', 'FeatureSet'])['FalsePositivesSyn'].sum() + \
                 df.groupby(['TargetModel', 'FeatureSet'])['FalseNegativesSyn'].sum()
    SuccessRateSyn = MatchesSyn / TotalSyn
    print(f'Prediction accuracy ALL (1)\n{SuccessRateSynTotal}')
    # 2. Mean/STD from different iterations
    SuccessRateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['AccuracySyn']. \
        agg({'AccuracySyn': {'Mean': np.mean, 'SD': np.std}})
    print(f'Prediction accuracy ALL (2)\n{SuccessRateSyn2}')

    # True positive rate (Recall)
    # 1. Summing all iterations and targets
    TruePositivesSyn = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesSyn'].sum()
    PositiveLabelsSyn = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesSyn'].sum() + \
                        df.groupby(['TargetModel', 'FeatureSet'])['FalseNegativesSyn'].sum()
    TPRateSyn = TruePositivesSyn / PositiveLabelsSyn
    print(f'True positive rate (Recall) ALL (1)\n{TPRateSyn}')
    # 2. Mean/STD from different iterations
    TPRateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['TPRateSyn']. \
        agg({'TPRateSyn': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) ALL (2)\n{TPRateSyn2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations and targets
    PositivesSyn = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesSyn'].sum() + \
                   df.groupby(['TargetModel', 'FeatureSet'])['FalsePositivesSyn'].sum()
    PPVRateSyn = TruePositivesSyn / PositivesSyn
    print(f'Positive Predictive Value (Precision) ALL (1)\n{PPVRateSyn}')
    # 2. Mean/STD from different iterations
    PPVRateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['PPVRateSyn']. \
        agg({'PPVRateSyn': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) ALL (2)\n{PPVRateSyn2}')

    # F1 rate
    # 1. Summing all iterations and targets
    F1RateSyn = 2 * (PPVRateSyn * TPRateSyn) / (PPVRateSyn + TPRateSyn)
    print(f'F1 rate ALL (1)\n{F1RateSyn}')
    # 2. Mean/STD from different iterations
    F1RateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['F1RateSyn']. \
        agg({'F1RateSyn': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate ALL (2)\n{F1RateSyn2}')

    # Raw
    # Prediction accuracy
    # 1. Summing all iterations and targets
    MatchesRaw = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesRaw'].sum() + \
                 df.groupby(['TargetModel', 'FeatureSet'])['TrueNegativesRaw'].sum()
    TotalRaw = MatchesRaw + \
               df.groupby(['TargetModel', 'FeatureSet'])['FalsePositivesRaw'].sum() + \
               df.groupby(['TargetModel', 'FeatureSet'])['FalseNegativesRaw'].sum()
    SuccessRateRaw = MatchesRaw / TotalRaw
    print(f'Prediction accuracy ALL (1)\n{SuccessRateRawTotal}')
    # 2. Mean/STD from different iterations
    SuccessRateRaw2 = df.groupby(['TargetModel', 'FeatureSet'])['AccuracyRaw']. \
        agg({'AccuracyRaw': {'Mean': np.mean, 'SD': np.std}})
    print(f'Prediction accuracy ALL (2)\n{SuccessRateRaw2}')

    # True positive rate (Recall)
    # 1. Summing all iterations and targets
    TruePositivesRaw = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesRaw'].sum()
    PositiveLabelsRaw = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesRaw'].sum() + \
                        df.groupby(['TargetModel', 'FeatureSet'])['FalseNegativesRaw'].sum()
    TPRateRaw = TruePositivesRaw / PositiveLabelsRaw
    print(f'True positive rate (Recall) ALL (1)\n{TPRateRaw}')
    # 2. Mean/STD from different iterations
    TPRateRaw2 = df.groupby(['TargetModel', 'FeatureSet'])['TPRateRaw']. \
        agg({'TPRateRaw': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) ALL (2)\n{TPRateRaw2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations and targets
    PositivesRaw = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesRaw'].sum() + \
                   df.groupby(['TargetModel', 'FeatureSet'])['FalsePositivesRaw'].sum()
    PPVRateRaw = TruePositivesRaw / PositivesRaw
    print(f'Positive Predictive Value (Precision) ALL (1)\n{PPVRateRaw}')
    # 2. Mean/STD from different iterations
    PPVRateRaw2 = df.groupby(['TargetModel', 'FeatureSet'])['PPVRateRaw']. \
        agg({'PPVRateRaw': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) ALL (2)\n{PPVRateRaw2}')

    # F1 rate
    # 1. Summing all iterations and targets
    F1RateRaw = 2 * (PPVRateRaw * TPRateRaw) / (PPVRateRaw + TPRateRaw)
    print(f'F1 rate ALL (1)\n{F1RateRaw}')
    # 2. Mean/STD from different iterations
    F1RateRaw2 = df.groupby(['TargetModel', 'FeatureSet'])['F1RateRaw']. \
        agg({'F1RateRaw': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate ALL (2)\n{F1RateRaw2}')

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
    print("\n\n\n\n\nUtility results")

    df = load_results_utility("tests/utility")[1]

    # All methods (Synthetic, raw, sanitised)

    Means = {var[5:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].
             agg({var[5:]: {'Mean': np.mean, 'SD': np.std}})
             for var in [c for c in df.columns if c.startswith('Mean_')]}
    print(f"Variable means:\n{Means}")
    Medians = {var[7:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].
               agg({var[7:]: {'Mean': np.mean, 'SD': np.std}})
               for var in [c for c in df.columns if c.startswith('Median_')]}
    print(f"Variable medians:\n{Medians}")
    for c in df.columns:
        if c.startswith('Freq_'):
            df[c] = df[c].apply(np.array)
    #df.loc[:,[c for c in df.columns if c.startswith('Freq_')]].apply(np.array) #= np.array(df[[c for c in df.columns if c.startswith('Freq_')]])
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

    print(f"Variable frequencies means:\n{Frequencies_Means}")
    print(f"Variable frequencies SD:\n{Frequencies_SDs}")

    #{k: np.std((v.values), axis=0) for k, v in Frequencies.items()}
    #np.std(tuple(df.iloc[0:4, 30].apply(np.array)), axis=0)
    #np.std(tuple(df.iloc[0:4, 30].apply(np.array)), axis=0)
    #np.std(tuple(df.iloc[0:4, 30].apply(np.array)), axis=0)

    Accuracy = df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])['Accuracy']. \
               agg({'Accuracy': {'Mean': np.mean, 'SD': np.std}})
    print(f"Classification accuracy:\n{Accuracy}")

    F1 = df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])['F1']. \
        agg({'F1': {'Mean': np.mean, 'SD': np.std}})
    print(f"Classification F1:\n{F1}")

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

    all_metrics = {
        "inference_per_target": inference_per_target,
        "inference_overall": inference_per_target,
        "linkage_per_target": linkage_per_target,
        "linkage_overall": linkage_overall,
        "utility_classification_overall": utility_classification_overall,
        "utility_means_overall": utility_means_overall,
        "utility_medians_overall": utility_medians_overall,
        "utility_freq_overall": utility_freq_overall}


if __name__ == "__main__":
    main()
