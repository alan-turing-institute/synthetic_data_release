"""
Command-line interface for running utility evaluation
"""

import json

from os import mkdir, path
from numpy import mean
from numpy.random import choice, seed
from argparse import ArgumentParser

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER

from sanitisation_techniques.sanitiser import SanitiserNHS
from generative_models.data_synthesiser import BayesianNet, PrivBayes, IndependentHistogram
from generative_models.ctgan import CTGAN
from generative_models.pate_gan import PATEGAN
from predictive_models.predictive_model import RandForestClassTask, LogRegClassTask, LinRegTask
from dython.nominal import associations

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='outputs/test', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    seed(SEED)
    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    if args.s3name is not None:
        rawPop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.split('/')[-1]

    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # make list of categorical/ordinal variables
    categorical_variables = [v['name'] for v in metadata['columns'] if v['type'] in ['Categorical', 'Ordinal']]

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    ########################
    #### GAME INPUTS #######
    ########################
    # Train test split - if neither 'dataFilter' or 'train_fraction' are
    # in the json file, KeyError is thrown
    # When using 'train_fraction', 'Targets' and 'TestRecords' should be left empty
    if runconfig.get('dataFilter') is None:
        rawTrain = rawPop.sample(frac=runconfig['train_fraction'])
        rawTest = rawPop.loc[~rawPop.index.isin(rawTrain.index)]
    else:
        rawTrain = rawPop.query(runconfig['dataFilter']['train'])
        rawTest = rawPop.query(runconfig['dataFilter']['test'])

    # Pick targets
    targetIDs = choice(list(rawTrain.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    targets = rawTrain.loc[targetIDs, :]

    # Drop targets from population
    rawTrainWoTargets = rawTrain.drop(targetIDs)

    # Get test target records
    testRecordIDs = choice(list(rawTest.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['TestRecords'] is not None:
        testRecordIDs.extend(runconfig['TestRecords'])

    testRecords = rawTest.loc[testRecordIDs, :]

    # List of candidate generative models to evaluate
    gmList = []
    if 'generativeModels' in runconfig.keys():
        for gm, paramsList in runconfig['generativeModels'].items():
            if gm == 'IndependentHistogram':
                for params in paramsList:
                    gmList.append(IndependentHistogram(metadata, *params))
            elif gm == 'BayesianNet':
                for params in paramsList:
                    gmList.append(BayesianNet(metadata, *params))
            elif gm == 'PrivBayes':
                for params in paramsList:
                    gmList.append(PrivBayes(metadata, *params))
            elif gm == 'CTGAN':
                for params in paramsList:
                    gmList.append(CTGAN(metadata, *params))
            elif gm == 'PATEGAN':
                for params in paramsList:
                    gmList.append(PATEGAN(metadata, *params))
            else:
                raise ValueError(f'Unknown GM {gm}')

    # List of candidate sanitisation techniques to evaluate
    sanList = []
    if 'sanitisationTechniques' in runconfig.keys():
        for name, paramsList in runconfig['sanitisationTechniques'].items():
            if name == 'SanitiserNHS':
                for params in paramsList:
                    sanList.append(SanitiserNHS(metadata, *params))
            else:
                raise ValueError(f'Unknown sanitisation technique {name}')

    utilityTasks = []
    for taskName, paramsList in runconfig['utilityTasks'].items():
        if taskName == 'RandForestClass':
            for params in paramsList:
                utilityTasks.append(RandForestClassTask(metadata, *params))
        elif taskName == 'LogRegClass':
            for params in paramsList:
                utilityTasks.append(LogRegClassTask(metadata, *params))
        elif taskName == 'LinReg':
            for params in paramsList:
                utilityTasks.append(LinRegTask(metadata, *params))

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsTargetUtility = {ut.__name__: {gm.__name__: {} for gm in gmList + sanList} for ut in utilityTasks}
    resultsAggUtility = {ut.__name__: {gm.__name__: {'TargetID': [],
                                                     'Accuracy': [],
                                                     'F1': [],
                                                     'VariableMeasures': {'Means': [], 'Medians': [],
                                                                          'Frequencies': [], 'Correlations': []}}
                                       for gm in gmList + sanList} for ut in utilityTasks}

    # Add entry for raw
    for ut in utilityTasks:
        resultsTargetUtility[ut.__name__]['Raw'] = {}
        resultsAggUtility[ut.__name__]['Raw'] = {'TargetID': [],
                                                 'Accuracy': [],
                                                 'F1': [],
                                                 'VariableMeasures': {'Means': [], 'Medians': [],
                                                                      'Frequencies': [], 'Correlations': []}}

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        # Draw a raw dataset
        rIdx = choice(list(rawTrainWoTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawTrain.loc[rIdx]

        LOGGER.info('Start: Utility evaluation on Raw...')
        # Get utility from raw without targets
        for ut in utilityTasks:
            resultsTargetUtility[ut.__name__]['Raw'][nr] = {}

            predErrorTargets = []
            predErrorAggr = []
            predF1Aggr = []
            for _ in range(runconfig['nSynT']):
                ut.train(rawTout)
                predErrorTargets.append(ut.evaluate(testRecords))
                predErrorAggr.append(ut.evaluate(rawTest))
                predF1Aggr.append(ut.f1(rawTest, positive_label=runconfig["positive_label"][ut.labelCol]))

            resultsTargetUtility[ut.__name__]['Raw'][nr]['OUT'] = {
                'TestRecordID': testRecordIDs,
                'Accuracy': list(mean(predErrorTargets, axis=0))
            }

            resultsAggUtility[ut.__name__]['Raw']['TargetID'].append('OUT')
            resultsAggUtility[ut.__name__]['Raw']['Accuracy'].append(mean(predErrorAggr))
            resultsAggUtility[ut.__name__]['Raw']['F1'].append(mean(predF1Aggr))
            resultsAggUtility[ut.__name__]['Raw']['VariableMeasures']["Means"].append(
                {
                i['name']: rawTout[i['name']].mean()
                for i in metadata["columns"]
                if i['name'] not in categorical_variables
                }
            )
            resultsAggUtility[ut.__name__]['Raw']['VariableMeasures']["Medians"].append(
                {
                i['name']: rawTout[i['name']].median()
                for i in metadata["columns"]
                if i['name'] not in categorical_variables
                }
            )
            resultsAggUtility[ut.__name__]['Raw']['VariableMeasures']["Frequencies"].append(
                {
                i['name']: rawTout[i['name']].value_counts(normalize=True).sort_index(ascending=True).tolist()
                for i in metadata["columns"]
                if i['name'] in categorical_variables
                }
            )
            # resultsAggUtility[ut.__name__]['Raw']['VariableMeasures']["Correlations"].append(
            #     associations(rawTout, nominal_columns=categorical_variables)
            # )

        # Get utility from raw with each target
        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawIn = rawTout.append(target)

            for ut in utilityTasks:
                predErrorTargets = []
                predErrorAggr = []
                predF1Aggr = []
                for _ in range(runconfig['nSynT']):
                    ut.train(rawIn)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))
                    predF1Aggr.append(ut.f1(rawTest, positive_label=runconfig["positive_label"][ut.labelCol]))

                resultsTargetUtility[ut.__name__]['Raw'][nr][tid] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }

                resultsAggUtility[ut.__name__]['Raw']['TargetID'].append(tid)
                resultsAggUtility[ut.__name__]['Raw']['Accuracy'].append(mean(predErrorAggr))
                resultsAggUtility[ut.__name__]['Raw']['F1'].append(mean(predF1Aggr))
                resultsAggUtility[ut.__name__]['Raw']['VariableMeasures']["Means"].append(
                    {
                        i['name']: rawIn[i['name']].mean()
                        for i in metadata["columns"]
                        if i['name'] not in categorical_variables
                    }
                )
                resultsAggUtility[ut.__name__]['Raw']['VariableMeasures']["Medians"].append(
                    {
                        i['name']: rawIn[i['name']].median()
                        for i in metadata["columns"]
                        if i['name'] not in categorical_variables
                    }
                )
                resultsAggUtility[ut.__name__]['Raw']['VariableMeasures']["Frequencies"].append(
                    {
                        i['name']: rawIn[i['name']].value_counts(normalize=True).sort_index(ascending=True).tolist()
                        for i in metadata["columns"]
                        if i['name'] in categorical_variables
                    }
                )
                # resultsAggUtility[ut.__name__]['Raw']['VariableMeasures']["Correlations"].append(
                #     associations(rawIn, nominal_columns=categorical_variables)
                # )

        LOGGER.info('Finished: Utility evaluation on Raw.')

        for GenModel in gmList:
            LOGGER.info(f'Start: Evaluation for model {GenModel.__name__}...')
            GenModel.fit(rawTout)
            synTwithoutTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]

            # Util evaluation for synthetic without all targets
            for ut in utilityTasks:
                resultsTargetUtility[ut.__name__][GenModel.__name__][nr] = {}

                predErrorTargets = []
                predErrorAggr = []
                predF1Aggr = []
                for syn in synTwithoutTarget:
                    ut.train(syn)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))
                    predF1Aggr.append(ut.f1(rawTest, positive_label=runconfig["positive_label"][ut.labelCol]))

                resultsTargetUtility[ut.__name__][GenModel.__name__][nr]['OUT'] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }

                resultsAggUtility[ut.__name__][GenModel.__name__]['TargetID'].append('OUT')
                resultsAggUtility[ut.__name__][GenModel.__name__]['Accuracy'].append(mean(predErrorAggr))
                resultsAggUtility[ut.__name__][GenModel.__name__]['F1'].append(mean(predF1Aggr))
                resultsAggUtility[ut.__name__][GenModel.__name__]['VariableMeasures']["Means"].append(
                    {
                        i['name']: syn[i['name']].mean()
                        for i in metadata["columns"]
                        if i['name'] not in categorical_variables
                    }
                )
                resultsAggUtility[ut.__name__][GenModel.__name__]['VariableMeasures']["Medians"].append(
                    {
                        i['name']: syn[i['name']].median()
                        for i in metadata["columns"]
                        if i['name'] not in categorical_variables
                    }
                )
                resultsAggUtility[ut.__name__][GenModel.__name__]['VariableMeasures']["Frequencies"].append(
                    {
                        i['name']: syn[i['name']].value_counts(normalize=True).sort_index(ascending=True).tolist()
                        for i in metadata["columns"]
                        if i['name'] in categorical_variables
                    }
                )
                # resultsAggUtility[ut.__name__][GenModel.__name__]['VariableMeasures']["Correlations"].append(
                #     associations(syn, nominal_columns=categorical_variables)
                # )

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]

                rawTin = rawTout.append(target)
                GenModel.fit(rawTin)
                synTwithTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]

                # Util evaluation for synthetic with this target
                for ut in utilityTasks:
                    predErrorTargets = []
                    predErrorAggr = []
                    predF1Aggr = []
                    for syn in synTwithTarget:
                        ut.train(syn)
                        predErrorTargets.append(ut.evaluate(testRecords))
                        predErrorAggr.append(ut.evaluate(rawTest))
                        predF1Aggr.append(ut.f1(rawTest, positive_label=runconfig["positive_label"][ut.labelCol]))

                    resultsTargetUtility[ut.__name__][GenModel.__name__][nr][tid] = {
                        'TestRecordID': testRecordIDs,
                        'Accuracy': list(mean(predErrorTargets, axis=0))
                    }

                    resultsAggUtility[ut.__name__][GenModel.__name__]['TargetID'].append(tid)
                    resultsAggUtility[ut.__name__][GenModel.__name__]['Accuracy'].append(mean(predErrorAggr))
                    resultsAggUtility[ut.__name__][GenModel.__name__]['F1'].append(mean(predF1Aggr))
                    resultsAggUtility[ut.__name__][GenModel.__name__]['VariableMeasures']["Means"].append(
                        {
                            i['name']: syn[i['name']].mean()
                            for i in metadata["columns"]
                            if i['name'] not in categorical_variables
                        }
                    )
                    resultsAggUtility[ut.__name__][GenModel.__name__]['VariableMeasures']["Medians"].append(
                        {
                            i['name']: syn[i['name']].median()
                            for i in metadata["columns"]
                            if i['name'] not in categorical_variables
                        }
                    )
                    resultsAggUtility[ut.__name__][GenModel.__name__]['VariableMeasures']["Frequencies"].append(
                        {
                            i['name']: syn[i['name']].value_counts(normalize=True).sort_index(ascending=True).tolist()
                            for i in metadata["columns"]
                            if i['name'] in categorical_variables
                        }
                    )
                    # resultsAggUtility[ut.__name__][GenModel.__name__]['VariableMeasures']["Correlations"].append(
                    #     associations(syn, nominal_columns=categorical_variables)
                    # )

            del synTwithoutTarget, synTwithTarget

            LOGGER.info(f'Finished: Evaluation for model {GenModel.__name__}.')

        for San in sanList:
            LOGGER.info(f'Start: Evaluation for sanitiser {San.__name__}...')
            sanOut = San.sanitise(rawTout)

            for ut in utilityTasks:
                resultsTargetUtility[ut.__name__][San.__name__][nr] = {}

                predErrorTargets = []
                predErrorAggr = []
                predF1Aggr = []
                for _ in range(runconfig['nSynT']):
                    ut.train(sanOut)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))
                    predF1Aggr.append(ut.f1(rawTest, positive_label=runconfig["positive_label"][ut.labelCol]))

                resultsTargetUtility[ut.__name__][San.__name__][nr]['OUT'] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }

                resultsAggUtility[ut.__name__][San.__name__]['TargetID'].append('OUT')
                resultsAggUtility[ut.__name__][San.__name__]['Accuracy'].append(mean(predErrorAggr))
                resultsAggUtility[ut.__name__][San.__name__]['F1'].append(mean(predF1Aggr))
                resultsAggUtility[ut.__name__][San.__name__]['VariableMeasures']["Means"].append(
                    {
                        i['name']: sanOut[i['name']].mean()
                        for i in metadata["columns"]
                        if i['name'] not in categorical_variables
                    }
                )
                resultsAggUtility[ut.__name__][San.__name__]['VariableMeasures']["Medians"].append(
                    {
                        i['name']: sanOut[i['name']].median()
                        for i in metadata["columns"]
                        if i['name'] not in categorical_variables
                    }
                )
                resultsAggUtility[ut.__name__][San.__name__]['VariableMeasures']["Frequencies"].append(
                    {
                        i['name']: sanOut[i['name']].value_counts(normalize=True).sort_index(ascending=True).tolist()
                        for i in metadata["columns"]
                        if i['name'] in categorical_variables
                    }
                )
                # resultsAggUtility[ut.__name__][San.__name__]['VariableMeasures']["Correlations"].append(
                #     associations(sanOut, nominal_columns=categorical_variables)
                # )

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]

                rawTin = rawTout.append(target)
                sanIn = San.sanitise(rawTin)

                for ut in utilityTasks:
                    predErrorTargets = []
                    predErrorAggr = []
                    predF1Aggr = []
                    for _ in range(runconfig['nSynT']):
                        ut.train(sanIn)
                        predErrorTargets.append(ut.evaluate(testRecords))
                        predErrorAggr.append(ut.evaluate(rawTest))
                        predF1Aggr.append(ut.f1(rawTest, positive_label=runconfig["positive_label"][ut.labelCol]))

                    resultsTargetUtility[ut.__name__][San.__name__][nr][tid] = {
                        'TestRecordID': testRecordIDs,
                        'Accuracy': list(mean(predErrorTargets, axis=0))
                    }

                    resultsAggUtility[ut.__name__][San.__name__]['TargetID'].append(tid)
                    resultsAggUtility[ut.__name__][San.__name__]['Accuracy'].append(mean(predErrorAggr))
                    resultsAggUtility[ut.__name__][San.__name__]['F1'].append(mean(predF1Aggr))
                    resultsAggUtility[ut.__name__][San.__name__]['VariableMeasures']["Means"].append(
                        {
                            i['name']: sanIn[i['name']].mean()
                            for i in metadata["columns"]
                            if i['name'] not in categorical_variables
                        }
                    )
                    resultsAggUtility[ut.__name__][San.__name__]['VariableMeasures']["Medians"].append(
                        {
                            i['name']: sanIn[i['name']].median()
                            for i in metadata["columns"]
                            if i['name'] not in categorical_variables
                        }
                    )
                    resultsAggUtility[ut.__name__][San.__name__]['VariableMeasures']["Frequencies"].append(
                        {
                            i['name']: sanIn[i['name']].value_counts(normalize=True).sort_index(ascending=True).tolist()
                            for i in metadata["columns"]
                            if i['name'] in categorical_variables
                        }
                    )
                    # resultsAggUtility[ut.__name__][San.__name__]['VariableMeasures']["Correlations"].append(
                    #     associations(sanIn, nominal_columns=categorical_variables)
                    # )

            del sanOut, sanIn

            LOGGER.info(f'Finished: Evaluation for model {San.__name__}.')

    outfile = f"ResultsUtilTargets_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsTargetUtility, f, indent=2, default=json_numpy_serialzer)

    outfile = f"ResultsUtilAgg_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsAggUtility, f, indent=2, default=json_numpy_serialzer)


if __name__ == "__main__":
    main()