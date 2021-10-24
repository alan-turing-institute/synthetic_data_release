import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from glob import glob
from pandas import DataFrame, concat
from itertools import cycle
from os import path
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')

from .datagen import load_local_data_as_df
from .plot_setup import set_style, pltmarkers as MARKERS, fontsizelabels as FSIZELABELS, fontsizeticks as FSIZETICKS
from .evaluation_framework import *
set_style()

PREDTASKS = ['RandomForestClassifier', 'LogisticRegression', 'LinearRegression']

MARKERCYCLE = cycle(MARKERS)
HUEMARKERS = [next(MARKERCYCLE) for _ in range(20)]


###### Load results
def load_results_linkage(dirname):
    """
    Helper function to load results of privacy evaluation under risk of linkability
    :param dirname: str: Directory that contains results files
    :return: results: DataFrame: Results of privacy evaluation
    """

    files = glob(path.join(dirname, f'ResultsMIA_*.json'))

    resList = []
    for fpath in files:
        with open(fpath) as f:
            resDict = json.load(f)

        dataset = fpath.split('.json')[0].split('_')[-1]

        for tid, tres in resDict.items():
            for gm, gmDict in tres.items():
                for nr, nrDict in gmDict.items():
                    for fset, fsetDict in nrDict.items():
                        df = DataFrame(fsetDict)

                        df['Run'] = nr
                        df['FeatureSet'] = fset
                        df['TargetModel'] = gm
                        df['TargetID'] = tid
                        df['Dataset'] = dataset

                        resList.append(df)

    results = concat(resList)

    resAgg = []

    games = results.groupby(['TargetID', 'TargetModel', 'FeatureSet', 'Run'])
    for gameParams, gameRes in games:
        tpSyn, fpSyn = get_tp_fp_rates(gameRes['AttackerGuess'], gameRes['Secret'])
        advantageSyn = get_mia_advantage(tpSyn, fpSyn)
        advantageRaw = 1

        tpS, tnS, fpS, fnS, \
        accS, TPrateS, PPVrateS, F1S = \
            get_rates_mem(gameRes['AttackerGuess'], gameRes['Secret'])

        tpR, tnR, fpR, fnR, \
        accR, TPrateR, PPVrateR, F1R = None, None, None, None, 1., 1., 1., 1.

        resAgg.append(gameParams + (tpSyn, fpSyn, advantageSyn, advantageRaw,
                                    tpR, tnR, fpR, fnR,
                                    tpS, tnS, fpS, fnS,
                                    accR, TPrateR, PPVrateR, F1R,
                                    accS, TPrateS, PPVrateS, F1S))

    resAgg = DataFrame(resAgg)

    resAgg.columns = ['TargetID', 'TargetModel', 'FeatureSet', 'Run',
                      'TPSyn', 'FPSyn', 'AdvantageSyn', 'AdvantageRaw',
                      'TruePositivesRaw', 'TrueNegativesRaw', 'FalsePositivesRaw', 'FalseNegativesRaw',
                      'TruePositivesSyn', 'TrueNegativesSyn', 'FalsePositivesSyn', 'FalseNegativesSyn',
                      'AccuracyRaw', 'TPRateRaw', 'PPVRateRaw', 'F1RateRaw',
                      'AccuracySyn', 'TPRateSyn', 'PPVRateSyn', 'F1RateSyn']

    resAgg['PrivacyGain'] = resAgg['AdvantageRaw'] - resAgg['AdvantageSyn']

    return resAgg


def load_results_inference(dirname, dpath, configpath):
    """
    Helper function to load results of privacy evaluation under risk of inference
    :param dirname: str: Directory that contains results files
    :param dpath: str: Dataset path (needed to extract some metadata)
    :return: results: DataFrame: Results of privacy evaluation
    """
    df, metadata = load_local_data_as_df(dpath)

    with open(f'{configpath}.json') as f:
        runconfig = json.load(f)

    files = glob(path.join(dirname, f'ResultsMLEAI_*.json'))
    resList = []
    for fpath in files:

        with open(fpath) as f:
            resDict = json.load(f)

        dataset = fpath.split('.json')[0].split('_')[-1]

        for tid, tdict in resDict.items():
            for sa, sdict in tdict.items():
                tsecret = df.loc[tid, sa]
                satype = None

                for cdict in metadata['columns']:
                    if cdict['name'] == sa:
                        satype = cdict['type']

                if '_' in sa:
                    sa = ''.join([s.capitalize() for s in sa.split('_')])
                elif '-' in sa:
                    sa = ''.join([s.capitalize() for s in sa.split('-')])

                for gm, gdict in sdict.items():
                    for nr, res in gdict.items():

                        resDF = DataFrame(res)
                        resDF['TargetID'] = tid
                        resDF['TargetSecret'] = tsecret
                        resDF['SensitiveType'] = satype
                        resDF['TargetModel'] = gm
                        resDF['Run'] = nr
                        resDF['SensitiveAttribute'] = sa
                        resDF['Dataset'] = dataset

                        resList.append(resDF)

    results = concat(resList)

    resAdv = []
    for gameParams, game in results.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'Run']):
        rawRes = game.groupby(['TargetModel']).get_group('Raw')
        if all(game['SensitiveType'].isin([INTEGER, FLOAT])):
            pCorrectRIn, pCorrectROut = get_probs_correct(rawRes['ProbCorrect'], rawRes['TargetPresence'])
            tpInR, tnInR, fpInR, fnInR, \
            tpOutR, tnOutR, fpOutR, fnOutR, \
            TPrateInR, TPrateOutR, TPrateTotalR, \
            PPVrateInR, PPVrateOutR, PPVrateTotalR, \
            F1InR, F1OutR, F1TotalR, AccInR, AccOutR, AccTotalR, F1AllR, F1macroAllR, AccAllR, AccBalAllR, TPAllR = \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                np.nan, np.nan

        elif all(game['SensitiveType'].isin([CATEGORICAL, ORDINAL])):
            pCorrectRIn, pCorrectROut = \
                get_accuracy(rawRes['AttackerGuess'], rawRes['TargetSecret'], rawRes['TargetPresence'])
            v = np.unique(rawRes["SensitiveAttribute"])
            if len(v) == 1:
                tpInR, tnInR, fpInR, fnInR, \
                tpOutR, tnOutR, fpOutR, fnOutR, \
                TPrateInR, TPrateOutR, TPrateTotalR, \
                PPVrateInR, PPVrateOutR, PPVrateTotalR, \
                F1InR, F1OutR, F1TotalR, AccInR, AccOutR, AccTotalR = \
                    get_rates_inf(rawRes['AttackerGuess'], rawRes['TargetSecret'],
                                 rawRes['TargetPresence'],
                                 runconfig["positive_label"][v[0]],
                                 runconfig["probIn"])
                F1AllR = mean(rawRes['GuessAllF1'][rawRes['TargetPresence'] == LABEL_OUT])
                F1macroAllR = mean(rawRes['GuessAllF1macro'][rawRes['TargetPresence'] == LABEL_OUT])
                AccAllR = mean(rawRes['GuessAllAcc'][rawRes['TargetPresence'] == LABEL_OUT])
                AccBalAllR = mean(rawRes['GuessAllAccBal'][rawRes['TargetPresence'] == LABEL_OUT])
                TPAllR = mean(rawRes['GuessAllTP'][rawRes['TargetPresence'] == LABEL_OUT])
            else:
                raise ValueError("More than one sensitive attribute in the same group")

        else:
            raise ValueError('Unknown sensitive attribute type.')

        advR = get_ai_advantage(pCorrectRIn, pCorrectROut)
        pSuccessR = get_prob_success_total(pCorrectRIn, pCorrectROut, runconfig["probIn"])

        for gm, gmRes in game.groupby(['TargetModel']):
            if gm != 'Raw':
                if all(gmRes['SensitiveType'].isin([INTEGER, FLOAT])):
                    pCorrectSIn, pCorrectSOut = get_probs_correct(gmRes['ProbCorrect'], gmRes['TargetPresence'])
                    tpInS, tnInS, fpInS, fnInS, \
                    tpOutS, tnOutS, fpOutS, fnOutS, \
                    TPrateInS, TPrateOutS, TPrateTotalS, \
                    PPVrateInS, PPVrateOutS, PPVrateTotalS, \
                    F1InS, F1OutS, F1TotalS, AccInS, AccOutS, AccTotalS, F1AllS, F1macroAllS, AccAllS, AccBalAllS, TPAllS = \
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                        np.nan, np.nan

                elif all(gmRes['SensitiveType'].isin([CATEGORICAL, ORDINAL])):
                    pCorrectSIn, pCorrectSOut = \
                        get_accuracy(gmRes['AttackerGuess'], gmRes['TargetSecret'], gmRes['TargetPresence'])
                    v = np.unique(gmRes["SensitiveAttribute"])
                    if len(v) == 1:
                        tpInS, tnInS, fpInS, fnInS, \
                        tpOutS, tnOutS, fpOutS, fnOutS, \
                        TPrateInS, TPrateOutS, TPrateTotalS, \
                        PPVrateInS, PPVrateOutS, PPVrateTotalS, \
                        F1InS, F1OutS, F1TotalS, AccInS, AccOutS, AccTotalS = \
                            get_rates_inf(gmRes['AttackerGuess'], gmRes['TargetSecret'],
                                         gmRes['TargetPresence'],
                                         runconfig["positive_label"][v[0]],
                                         runconfig["probIn"])
                        F1AllS = mean(gmRes['GuessAllF1'][gmRes['TargetPresence'] == LABEL_OUT])
                        F1macroAllS = mean(gmRes['GuessAllF1macro'][gmRes['TargetPresence'] == LABEL_OUT])
                        AccAllS = mean(gmRes['GuessAllAcc'][gmRes['TargetPresence'] == LABEL_OUT])
                        AccBalAllS = mean(gmRes['GuessAllAccBal'][gmRes['TargetPresence'] == LABEL_OUT])
                        TPAllS = mean(gmRes['GuessAllTP'][gmRes['TargetPresence'] == LABEL_OUT])
                    else:
                        raise ValueError("More than one sensitive attribute in the same group")
                else:
                    raise ValueError('Unknown sensitive attribute type.')

                advS = get_ai_advantage(pCorrectSIn, pCorrectSOut)
                pSuccessS = get_prob_success_total(pCorrectSIn, pCorrectSOut, runconfig["probIn"])

                resAdv.append(gameParams + (gm, pCorrectRIn, pCorrectROut, advR, pCorrectSIn, pCorrectSOut, advS,
                                            tpInR, tnInR, fpInR, fnInR, tpInS, tnInS, fpInS, fnInS,
                                            tpOutR, tnOutR, fpOutR, fnOutR, tpOutS, tnOutS, fpOutS, fnOutS,
                                            pSuccessR, pSuccessS,
                                            TPrateInR, TPrateOutR, TPrateTotalR, TPrateInS, TPrateOutS, TPrateTotalS,
                                            PPVrateInR, PPVrateOutR, PPVrateTotalR, PPVrateInS, PPVrateOutS, PPVrateTotalS,
                                            F1InR, F1OutR, F1TotalR, F1InS, F1OutS, F1TotalS,
                                            AccInR, AccOutR, AccTotalR, AccInS, AccOutS, AccTotalS,
                                            F1AllR, F1macroAllR, F1AllS, F1macroAllS,
                                            AccAllR, AccBalAllR, AccAllS, AccBalAllS, TPAllR, TPAllS))


    resAdv = DataFrame(resAdv)
    resAdv.columns = ['Dataset', 'TargetID', 'SensitiveAttribute', 'Run', 'TargetModel',
                      'ProbCorrectRawIn', 'ProbCorrectRawOut', 'AdvantageRaw',
                      'ProbCorrectSynIn', 'ProbCorrectSynOut', 'AdvantageSyn',
                      'TruePositivesRawIn', 'TrueNegativesRawIn', 'FalsePositivesRawIn', 'FalseNegativesRawIn',
                      'TruePositivesSynIn', 'TrueNegativesSynIn', 'FalsePositivesSynIn', 'FalseNegativesSynIn',
                      'TruePositivesRawOut', 'TrueNegativesRawOut', 'FalsePositivesRawOut', 'FalseNegativesRawOut',
                      'TruePositivesSynOut', 'TrueNegativesSynOut', 'FalsePositivesSynOut', 'FalseNegativesSynOut',
                      'ProbSuccessRaw', 'ProbSuccessSyn',
                      'TPRateRawIn', 'TPRateRawOut', 'TPRateRawTotal', 'TPRateSynIn', 'TPRateSynOut', 'TPRateSynTotal',
                      'PPVRateRawIn', 'PPVRateRawOut', 'PPVRateRawTotal', 'PPVRateSynIn', 'PPVRateSynOut', 'PPVRateSynTotal',
                      'F1RateRawIn', 'F1RateRawOut', 'F1RateRawTotal', 'F1RateSynIn', 'F1RateSynOut', 'F1RateSynTotal',
                      'AccRawIn', 'AccRawOut', 'AccRawTotal', 'AccSynIn', 'AccSynOut', 'AccSynTotal',
                      'F1RateRawAll', 'F1MacroRateRawAll', 'F1RateSynAll', 'F1MacroRateSynAll',
                      'AccRateRawAll', 'AccBalRateRawAll', 'AccRateSynAll', 'AccBalRateSynAll',
                      'TPRateRawAll', 'TPRateSynAll']

    resAdv['PrivacyGain'] = resAdv['AdvantageRaw'] - resAdv['AdvantageSyn']

    return resAdv


def load_results_utility(dirname):
    """
    Helper function to load results of utility evaluation
    :param dirname: str: Directory that contains results files
    :return: resultsTarget: DataFrame: Results of utility evaluation on individual records
    :return: resultsAgg: DataFrame: Results of average utility evaluation
    """

    # Load individual target utility results
    files = glob(path.join(dirname, f'ResultsUtilTargets_*.json'))

    resList = []
    for fpath in files:
        with open(fpath) as f:
            results = json.load(f)

        dataset = fpath.split('.json')[0].split('_')[-1]

        for ut, ures in results.items():
            model = [m for m in PREDTASKS if m in ut][0]
            labelVar = ut.split(model)[-1]

            if '_' in labelVar:
                labelVar = ''.join([s.capitalize() for s in labelVar.split('_')])

            if '-' in labelVar:
                labelVar = ''.join([s.capitalize() for s in labelVar.split('-')])

            for gm, gmres in ures.items():
                for n, nres in gmres.items():
                    for tid, tres in nres.items():
                        res = DataFrame(tres)

                        res['TargetID'] = tid
                        res['Run'] = f'Run {n}'
                        res['TargetModel'] = gm
                        res['PredictionModel'] = model
                        res['LabelVar'] = labelVar
                        res['Dataset'] = dataset

                        resList.append(res)

    resultsTargets = concat(resList)

    # Load aggregate utility results
    files = glob(path.join(dirname, f'ResultsUtilAgg_*.json'))

    resList = []
    for fpath in files:
        with open(fpath) as f:
            results = json.load(f)

        dataset = fpath.split('.json')[0].split('_')[-1]

        for ut, utres in results.items():
            model = [m for m in PREDTASKS if m in ut][0]
            labelVar = ut.split(model)[-1]

            if '_' in labelVar:
                labelVar = ''.join([s.capitalize() for s in labelVar.split('_')])

            if '-' in labelVar:
                labelVar = ''.join([s.capitalize() for s in labelVar.split('-')])

            for gm, gmres in utres.items():
                resDF = DataFrame({k: i for k, i in gmres.items() if k != 'VariableMeasures'})
                resDF['PredictionModel'] = model
                resDF['LabelVar'] = labelVar
                resDF['TargetModel'] = gm
                resDF['Dataset'] = dataset
                meansDF = pd.Series(gmres['VariableMeasures']['Means']).apply(pd.Series).add_prefix('Mean_')
                resDF = pd.concat([resDF, meansDF], axis=1)
                mediansDF = pd.Series(gmres['VariableMeasures']['Medians']).apply(pd.Series).add_prefix('Median_')
                resDF = pd.concat([resDF, mediansDF], axis=1)
                frequenciesDF = pd.Series(gmres['VariableMeasures']['Frequencies']).apply(pd.Series).add_prefix('Freq_')
                resDF = pd.concat([resDF, frequenciesDF], axis=1)

                resList.append(resDF)

    resultsAgg = concat(resList)

    return resultsTargets, resultsAgg


### Plotting
def plt_per_target_pg(results, models, resFilter=('FeatureSet', 'Naive')):
    """ Plot per record average privacy gain. """
    results = results[results[resFilter[0]] == resFilter[1]]

    fig, ax = plt.subplots()
    pointplot(results, 'TargetModel', 'PrivacyGain', 'TargetID', ax, models)

    ax.set_title(f'Attack on {resFilter[0]}: {resFilter[1]}', fontsize=FSIZELABELS)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.3), ncol=5, title='TargetID')
    ax.set_ylabel('$\mathtt{PG}$', fontsize=FSIZELABELS)

    return fig


def pointplot(data, x, y, hue, ax, order):
    ncats = data[hue].nunique()
    huemarkers = HUEMARKERS[:ncats]

    sns.pointplot(data=data, y=y,
                  x=x, hue=hue,
                  order=order,
                  ax=ax, dodge=True,
                  join=False, markers=huemarkers,
                  scale=1.2, errwidth=2,
                  linestyles='--')

    # Remove legend
    ax.get_legend().remove()

    # Set x- and y-label
    ax.set_xlabel('')

    # Resize y-tick labels
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    # Resize x-tick labels
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)
