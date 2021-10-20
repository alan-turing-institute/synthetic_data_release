import json
import numpy as np
from utils.analyse_results import load_results_inference, load_results_linkage, load_results_utility
from argparse import ArgumentParser


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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) (2)\n{PPVRateSynTotal2}')

    # F1 rate
    # 1. Summing all iterations
    F1RateSynIn = 2 * (PPVRateSynIn * TPRateSynIn) / (PPVRateSynIn + TPRateSynIn)
    F1RateSynOut = 2 * (PPVRateSynOut * TPRateSynOut) / (PPVRateSynOut + TPRateSynOut)
    F1RateSynTotal = F1RateSynIn * pIn + F1RateSynOut * (1 - pIn)
    print(f'F1 rate (1)\n{F1RateSynTotal}')
    # 2. Mean/STD from different iterations
    F1RateSynTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['F1RateSynTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate (2)\n{F1RateSynTotal2}')

    # For all targets combined
    print("For all targets combined:")

    # Prediction accuracy
    # 1. Summing all iterations and targets
    MatchesSynIn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum() + \
                   df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesSynIn'].sum()
    TotalSynIn = MatchesSynIn + \
                 df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynIn'].sum() + \
                 df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynIn'].sum()
    SuccessRateSynIn = MatchesSynIn / TotalSynIn
    MatchesSynOut = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum() + \
                   df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesSynOut'].sum()
    TotalSynOut = MatchesSynOut + \
                 df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynOut'].sum() + \
                 df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynOut'].sum()
    SuccessRateSynOut = MatchesSynOut / TotalSynOut
    SuccessRateSynTotal = SuccessRateSynIn * pIn + SuccessRateSynOut * (1 - pIn)
    print(f'Prediction accuracy ALL (1)\n{SuccessRateSynTotal}')
    # 2. Mean/STD from different iterations
    SuccessRateSynTotal2 = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccSynTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'Prediction accuracy ALL (2)\n{SuccessRateSynTotal2}')

    # True positive rate (Recall)
    # 1. Summing all iterations and targets
    TruePositivesSynIn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum()
    PositiveLabelsSynIn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum() + \
                          df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynIn'].sum()
    TPRateSynIn = TruePositivesSynIn / PositiveLabelsSynIn
    TruePositivesSynOut = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum()
    PositiveLabelsSynOut = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum() + \
                          df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynOut'].sum()
    TPRateSynOut = TruePositivesSynOut / PositiveLabelsSynOut
    TPRateSynTotal = TPRateSynIn * pIn + TPRateSynOut * (1 - pIn)
    print(f'True positive rate (Recall) ALL (1)\n{TPRateSynTotal}')
    # 2. Mean/STD from different iterations
    TPRateSynTotal2 = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TPRateSynTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) ALL (2)\n{TPRateSynTotal2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations and targets
    PositivesSynIn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum() + \
                     df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynIn'].sum()
    PPVRateSynIn = TruePositivesSynIn / PositivesSynIn
    PositivesSynOut = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum() + \
                     df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynOut'].sum()
    PPVRateSynOut = TruePositivesSynOut / PositivesSynOut
    PPVRateSynTotal = PPVRateSynIn * pIn + PPVRateSynOut * (1 - pIn)
    print(f'Positive Predictive Value (Precision) ALL (1)\n{PPVRateSynTotal}')
    # 2. Mean/STD from different iterations
    PPVRateSynTotal2 = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['PPVRateSynTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) ALL (2)\n{PPVRateSynTotal2}')

    # F1 rate
    # 1. Summing all iterations and targets
    F1RateSynIn = 2 * (PPVRateSynIn * TPRateSynIn) / (PPVRateSynIn + TPRateSynIn)
    F1RateSynOut = 2 * (PPVRateSynOut * TPRateSynOut) / (PPVRateSynOut + TPRateSynOut)
    F1RateSynTotal = F1RateSynIn * pIn + F1RateSynOut * (1 - pIn)
    print(f'F1 rate ALL (1)\n{F1RateSynTotal}')
    # 2. Mean/STD from different iterations
    F1RateSynTotal2 = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1RateSynTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate ALL (2)\n{F1RateSynTotal2}')

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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) (2)\n{PPVRateRawTotal2}')

    # F1 rate
    # 1. Summing all iterations
    F1RateRawIn = 2 * (PPVRateRawIn * TPRateRawIn) / (PPVRateRawIn + TPRateRawIn)
    F1RateRawOut = 2 * (PPVRateRawOut * TPRateRawOut) / (PPVRateRawOut + TPRateRawOut)
    F1RateRawTotal = F1RateRawIn * pIn + F1RateRawOut * (1 - pIn)
    print(f'F1 rate (1)\n{F1RateRawTotal}')
    # 2. Mean/STD from different iterations
    F1RateRawTotal2 = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['F1RateRawTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate (2)\n{F1RateRawTotal2}')

    # For all targets combined
    print("For all targets combined:")

    # Prediction accuracy
    # 1. Summing all iterations and targets
    MatchesRawIn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawIn'].sum() + \
                   df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesRawIn'].sum()
    TotalRawIn = MatchesRawIn + \
                 df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawIn'].sum() + \
                 df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawIn'].sum()
    SuccessRateRawIn = MatchesRawIn / TotalRawIn
    MatchesRawOut = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawOut'].sum() + \
                    df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesRawOut'].sum()
    TotalRawOut = MatchesRawOut + \
                  df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawOut'].sum() + \
                  df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawOut'].sum()
    SuccessRateRawOut = MatchesRawOut / TotalRawOut
    SuccessRateRawTotal = SuccessRateRawIn * pIn + SuccessRateRawOut * (1 - pIn)
    print(f'Prediction accuracy ALL (1)\n{SuccessRateRawTotal}')
    # 2. Mean/STD from different iterations
    SuccessRateRawTotal2 = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['AccRawTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'Prediction accuracy ALL (2)\n{SuccessRateRawTotal2}')

    # True positive rate (Recall)
    # 1. Summing all iterations and targets
    TruePositivesRawIn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawIn'].sum()
    PositiveLabelsRawIn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawIn'].sum() + \
                          df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawIn'].sum()
    TPRateRawIn = TruePositivesRawIn / PositiveLabelsRawIn
    TruePositivesRawOut = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawOut'].sum()
    PositiveLabelsRawOut = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawOut'].sum() + \
                           df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawOut'].sum()
    TPRateRawOut = TruePositivesRawOut / PositiveLabelsRawOut
    TPRateRawTotal = TPRateRawIn * pIn + TPRateRawOut * (1 - pIn)
    print(f'True positive rate (Recall) ALL (1)\n{TPRateRawTotal}')
    # 2. Mean/STD from different iterations
    TPRateRawTotal2 = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TPRateRawTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) ALL (2)\n{TPRateRawTotal2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations and targets
    PositivesRawIn = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawIn'].sum() + \
                     df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawIn'].sum()
    PPVRateRawIn = TruePositivesRawIn / PositivesRawIn
    PositivesRawOut = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawOut'].sum() + \
                      df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawOut'].sum()
    PPVRateRawOut = TruePositivesRawOut / PositivesRawOut
    PPVRateRawTotal = PPVRateRawIn * pIn + PPVRateRawOut * (1 - pIn)
    print(f'Positive Predictive Value (Precision) ALL (1)\n{PPVRateRawTotal}')
    # 2. Mean/STD from different iterations
    PPVRateRawTotal2 = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['PPVRateRawTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) ALL (2)\n{PPVRateRawTotal2}')

    # F1 rate
    # 1. Summing all iterations and targets
    F1RateRawIn = 2 * (PPVRateRawIn * TPRateRawIn) / (PPVRateRawIn + TPRateRawIn)
    F1RateRawOut = 2 * (PPVRateRawOut * TPRateRawOut) / (PPVRateRawOut + TPRateRawOut)
    F1RateRawTotal = F1RateRawIn * pIn + F1RateRawOut * (1 - pIn)
    print(f'F1 rate ALL (1)\n{F1RateRawTotal}')
    # 2. Mean/STD from different iterations
    F1RateRawTotal2 = df.groupby(['Dataset', 'SensitiveAttribute', 'TargetModel'])['F1RateRawTotal']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate ALL (2)\n{F1RateRawTotal2}')




    # Linkage attack
    print("Linkage attack results")

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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) (2)\n{PPVRateSyn2}')

    # F1 rate
    # 1. Summing all iterations
    F1RateSyn = 2 * (PPVRateSyn * TPRateSyn) / (PPVRateSyn + TPRateSyn)
    print(f'F1 rate (1)\n{F1RateSyn}')
    # 2. Mean/STD from different iterations
    F1RateSyn2 = df.groupby(['TargetID', 'TargetModel', 'FeatureSet'])['F1RateSyn']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate (2)\n{F1RateSyn2}')

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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
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
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'True positive rate (Recall) ALL (2)\n{TPRateSyn2}')

    # Positive Predictive Value (Precision)
    # 1. Summing all iterations and targets
    PositivesSyn = df.groupby(['TargetModel', 'FeatureSet'])['TruePositivesSyn'].sum() + \
                   df.groupby(['TargetModel', 'FeatureSet'])['FalsePositivesSyn'].sum()
    PPVRateSyn = TruePositivesSyn / PositivesSyn
    print(f'Positive Predictive Value (Precision) ALL (1)\n{PPVRateSyn}')
    # 2. Mean/STD from different iterations
    PPVRateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['PPVRateSyn']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'Positive Predictive Value (Precision) ALL (2)\n{PPVRateSyn2}')

    # F1 rate
    # 1. Summing all iterations and targets
    F1RateSyn = 2 * (PPVRateSyn * TPRateSyn) / (PPVRateSyn + TPRateSyn)
    print(f'F1 rate ALL (1)\n{F1RateSyn}')
    # 2. Mean/STD from different iterations
    F1RateSyn2 = df.groupby(['TargetModel', 'FeatureSet'])['F1RateSyn']. \
        agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f'F1 rate ALL (2)\n{F1RateSyn2}')
    
    
    
    
    
    # Utility
    print("Utility results")

    df = load_results_utility("tests/utility")[1]

    # All methods (Synthetic, raw, sanitised)

    Means = {var[5:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].
             agg({'Score': {'Mean': np.mean, 'SD': np.std}})
             for var in [c for c in df.columns if c.startswith('Mean_')]}
    print(f"Variable means:\n{Means}")
    Medians = {var[7:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].
               agg({'Score': {'Mean': np.mean, 'SD': np.std}})
               for var in [c for c in df.columns if c.startswith('Median_')]}
    print(f"Variable medians:\n{Medians}")
    for c in df.columns:
        if c.startswith('Freq_'):
            df[c] = df[c].apply(np.array)
    #df.loc[:,[c for c in df.columns if c.startswith('Freq_')]].apply(np.array) #= np.array(df[[c for c in df.columns if c.startswith('Freq_')]])
    Frequencies_Means = {
        var[5:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].apply(np.array)
        for var in [c for c in df.columns if c.startswith('Freq_')]}
    Frequencies_SDs = {
        var[5:]: df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])[var].apply(np.array)
        for var in [c for c in df.columns if c.startswith('Freq_')]}

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

    for k in Frequencies_Means.keys():
        try:
            Frequencies_Means[k] = Frequencies_Means[k].apply(custom_mean)
        except ValueError:
            Frequencies_Means[k] = np.nan
    for k in Frequencies_SDs.keys():
        try:
            Frequencies_SDs[k] = Frequencies_SDs[k].apply(custom_std)
        except ValueError:
            Frequencies_SDs[k] = np.nan

    print(f"Variable frequencies means:\n{Frequencies_Means}")
    print(f"Variable frequencies means:\n{Frequencies_SDs}")

    #{k: np.std((v.values), axis=0) for k, v in Frequencies.items()}
    #np.std(tuple(df.iloc[0:4, 30].apply(np.array)), axis=0)
    #np.std(tuple(df.iloc[0:4, 30].apply(np.array)), axis=0)
    #np.std(tuple(df.iloc[0:4, 30].apply(np.array)), axis=0)

    Accuracy = df.groupby(['Dataset', 'TargetModel', 'PredictionModel', 'LabelVar'])['Accuracy']. \
               agg({'Score': {'Mean': np.mean, 'SD': np.std}})
    print(f"Classification accuracy:\n{Accuracy}")


if __name__ == "__main__":
    main()
