import json
from utils.analyse_results import load_results_inference


# Inference attack
print("Inference attack")

df = load_results_inference("tests/inference", "data/texas", "tests/inference/runconfig")
with open(f'tests/inference/runconfig_cchic.json') as f:
    runconfig = json.load(f)
pIn = runconfig['probIn']

# Synthetic results
print("Synthetic dataset")

# Prediction accuracy
MatchesInSyn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['MatchesInSyn'].sum()
TotalInSyn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TotalInSyn'].sum()
MatchesOutSyn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['MatchesOutSyn'].sum()
TotalOutSyn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TotalOutSyn'].sum()
SuccessRateInSyn = MatchesInSyn / TotalInSyn
SuccessRateOutSyn = MatchesOutSyn / TotalOutSyn
SuccessRateTotalSyn = SuccessRateInSyn * pIn + SuccessRateOutSyn * (1 - pIn)
print(f'Prediction accuracy\n{SuccessRateTotalSyn}')

# True positive rate
TruePosInSyn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesInSyn'].sum()
PosInSyn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['PositivesInSyn'].sum()
TruePosOutSyn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesOutSyn'].sum()
PosOutSyn = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['PositivesOutSyn'].sum()
TPrateInSyn = TruePosInSyn / PosInSyn
TPrateOutSyn = TruePosOutSyn / PosOutSyn
TPrateTotalSyn = TPrateInSyn * pIn + TPrateOutSyn * (1 - pIn)
print(f'True positive rate\n{TPrateTotalSyn}')


# Raw results
print("Raw dataset")

# Prediction accuracy
MatchesInRaw = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['MatchesInRaw'].sum()
TotalInRaw = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TotalInRaw'].sum()
MatchesOutRaw = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['MatchesOutRaw'].sum()
TotalOutRaw = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TotalOutRaw'].sum()
SuccessRateInRaw = MatchesInRaw / TotalInRaw
SuccessRateOutRaw = MatchesOutRaw / TotalOutRaw
SuccessRateTotalRaw = SuccessRateInRaw * pIn + SuccessRateOutRaw * (1 - pIn)
print(f'Prediction accuracy\n{SuccessRateTotalRaw}')

# True positive rate
TruePosInRaw = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesInRaw'].sum()
PosInRaw = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['PositivesInRaw'].sum()
TruePosOutRaw = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['TruePositivesOutRaw'].sum()
PosOutRaw = df.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'TargetModel'])['PositivesOutRaw'].sum()
TPrateInRaw = TruePosInRaw / PosInRaw
TPrateOutRaw = TruePosOutRaw / PosOutRaw
TPrateTotalRaw = TPrateInRaw * pIn + TPrateOutRaw * (1 - pIn)
print(f'True positive rate\n{TPrateTotalRaw}')