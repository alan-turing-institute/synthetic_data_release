import json
from utils.analyse_results import load_results_inference

# Inference attack
print("Inference attack results")
dataset = "data/cchic_cleaned"
config_file = "tests/inference/runconfig_cchic"

df = load_results_inference("tests/inference", dataset, config_file)
with open(config_file) as f:
    runconfig = json.load(f)
pIn = runconfig['probIn']

# Synthetic results
print("Synthetic dataset")

# Prediction accuracy
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
print(f'Prediction accuracy\n{SuccessRateSynTotal}')

# True positive rate (Recall)
TruePositivesSynIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum()
PositiveLabelsSynIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum() + \
                      df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynIn'].sum()
TPRateSynIn = TruePositivesSynIn / PositiveLabelsSynIn
TruePositivesSynOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum()
PositiveLabelsSynOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum() + \
                      df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesSynOut'].sum()
TPRateSynOut = TruePositivesSynOut / PositiveLabelsSynOut
TPRateSynTotal = TPRateSynIn * pIn + TPRateSynOut * (1 - pIn)
print(f'True positive rate (Recall)\n{TPRateSynTotal}')

# Positive Predictive Value (Precision)
PositivesSynIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynIn'].sum() + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynIn'].sum()
PPVRateSynIn = TruePositivesSynIn / PositivesSynIn
PositivesSynOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesSynOut'].sum() + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesSynOut'].sum()
PPVRateSynOut = TruePositivesSynOut / PositivesSynOut
PPVRateSynTotal = PPVRateSynIn * pIn + PPVRateSynOut * (1 - pIn)
print(f'Positive Predictive Value (Precision)\n{PPVRateSynTotal}')

# F1 rate
F1RateSynIn = 2 * (PPVRateSynIn * TPRateSynIn) / (PPVRateSynIn + TPRateSynIn)
F1RateSynOut = 2 * (PPVRateSynOut * TPRateSynOut) / (PPVRateSynOut + TPRateSynOut)
F1RateSynTotal = F1RateSynIn * pIn + F1RateSynOut * (1 - pIn)
print(f'F1 rate\n{F1RateSynTotal}')


# Raw results
print("Raw dataset")

# Prediction accuracy
MatchesRawIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawIn'].sum() + \
               df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesRawIn'].sum()
TotalRawIn = MatchesRawIn + \
             df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawIn'].sum() + \
             df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawIn'].sum()
SuccessRateRawIn = MatchesRawIn / TotalRawIn
MatchesRawOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawOut'].sum() + \
               df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TrueNegativesRawOut'].sum()
TotalRawOut = MatchesRawOut + \
             df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawOut'].sum() + \
             df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawOut'].sum()
SuccessRateRawOut = MatchesRawOut / TotalRawOut
SuccessRateRawTotal = SuccessRateRawIn * pIn + SuccessRateRawOut * (1 - pIn)
print(f'Prediction accuracy\n{SuccessRateRawTotal}')

# True positive rate (Recall)
TruePositivesRawIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawIn'].sum()
PositiveLabelsRawIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawIn'].sum() + \
                      df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawIn'].sum()
TPRateRawIn = TruePositivesRawIn / PositiveLabelsRawIn
TruePositivesRawOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawOut'].sum()
PositiveLabelsRawOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawOut'].sum() + \
                      df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalseNegativesRawOut'].sum()
TPRateRawOut = TruePositivesRawOut / PositiveLabelsRawOut
TPRateRawTotal = TPRateRawIn * pIn + TPRateRawOut * (1 - pIn)
print(f'True positive rate (Recall)\n{TPRateRawTotal}')

# Positive Predictive Value (Precision)
PositivesRawIn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawIn'].sum() + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawIn'].sum()
PPVRateRawIn = TruePositivesRawIn / PositivesRawIn
PositivesRawOut = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesRawOut'].sum() + \
                 df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['FalsePositivesRawOut'].sum()
PPVRateRawOut = TruePositivesRawOut / PositivesRawOut
PPVRateRawTotal = PPVRateRawIn * pIn + PPVRateRawOut * (1 - pIn)
print(f'Positive Predictive Value (Precision)\n{PPVRateRawTotal}')

# F1 rate
F1RateRawIn = 2 * (PPVRateRawIn * TPRateRawIn) / (PPVRateRawIn + TPRateRawIn)
F1RateRawOut = 2 * (PPVRateRawOut * TPRateRawOut) / (PPVRateRawOut + TPRateRawOut)
F1RateRawTotal = F1RateRawIn * pIn + F1RateRawOut * (1 - pIn)
print(f'F1 rate\n{F1RateRawTotal}')