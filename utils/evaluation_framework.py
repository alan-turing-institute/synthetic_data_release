"""
Procedures for running a privacy evaluation on a generative model
"""

import numpy as np
from numpy import where, mean, nan
from sklearn.metrics import f1_score, confusion_matrix

from utils.constants import *


def get_accuracy(guesses, labels, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    pIn = sum([g == l for g, l in zip(guesses[idxIn], labels[idxIn])]) / len(idxIn)
    pOut = sum([g == l for g, l in zip(guesses[idxOut], labels[idxOut])]) / len(idxOut)

    return pIn, pOut


def get_tp_fp_rates(guesses, labels):
    targetIn = where(labels == LABEL_IN)[0]
    targetOut = where(labels == LABEL_OUT)[0]
    return sum(guesses[targetIn] == LABEL_IN) / len(targetIn), sum(guesses[targetOut] == LABEL_IN) / len(targetOut)


def get_probs_correct(pdf, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    pdf[pdf > 1.] = 1.
    return mean(pdf[idxIn]), mean(pdf[idxOut])


def get_mia_advantage(tp_rate, fp_rate):
    return tp_rate - fp_rate


def get_ai_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_util_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_prob_removed(before, after):
    idxIn = where(before == LABEL_IN)[0]
    return 1.0 - sum(after[idxIn] / len(idxIn))


def get_prob_success_total(pCorrectIn, pCorrectOut, probIn):
    """Calculate total probability of success by weighing the
    two scenarios of including or not including the target record
    in the training data."""
    return pCorrectIn * probIn + pCorrectOut * (1 - probIn)


def tpfp(predictions, ground_truth, positive):
    """
    Return accuracy and true/false negative/positive guesses.
    Args:
        predictions: an array of predicted labels
        ground_truth: an array of ground truth labels
        negative: a sentinel value indicating a negative label
        positive: a sentinel value indicating a positive label
    Returns:
        'acc': the binary classification accuracy
        'tp':  the amount of true positives
        'tn':  the amount of true negatives
        'fp':  the amount of false positives
        'fn':  the amount of false negatives
    """

    # compute the raw accuracy
    acc = np.mean(predictions == ground_truth)

    # accumulate the true/false negative/positives
    tp = np.sum(np.logical_and(predictions == positive, ground_truth == positive))
    tn = np.sum(np.logical_and(predictions != positive, ground_truth != positive))
    fp = np.sum(np.logical_and(predictions == positive, ground_truth != positive))
    fn = np.sum(np.logical_and(predictions != positive, ground_truth == positive))

    # return a dictionary of the raw accuracy and true/false positive/negative values
    return acc, tp, tn, fp, fn


def get_rates(guesses, labels, targetPresence, positive_label, probIn):
    """Calculate true and total positives, false positives, true positive rate
    (recall), positive predictive value (precision) and F1 rate for classification task"""

    # if label is in
    idxIn = where(targetPresence == LABEL_IN)[0]
    guessesIn = guesses[idxIn]
    labelsIn = labels[idxIn]

    accIn, tpIn, tnIn, fpIn, fnIn = tpfp(guessesIn, labelsIn, positive_label)

    # True positive rate (recall)
    if tpIn + fnIn > 0:
        TPrateIn = tpIn / (tpIn + fnIn)
    else:
        TPrateIn = nan

    # Positive predictive value (precision)
    if tpIn + fpIn > 0:
        PPVrateIn = tpIn / (tpIn + fpIn)
    else:
        PPVrateIn = nan

    # F1 rate
    if TPrateIn + PPVrateIn > 0:
        F1In = 2 * (TPrateIn * PPVrateIn) / (TPrateIn + PPVrateIn)
    else:
        F1In = nan

    # if label is out
    idxOut = where(targetPresence == LABEL_OUT)[0]
    guessesOut = guesses[idxOut]
    labelsOut = labels[idxOut]
    accOut, tpOut, tnOut, fpOut, fnOut = tpfp(guessesOut, labelsOut, positive_label)

    # True positive rate (recall)
    if tpOut + fnOut > 0:
        TPrateOut = tpOut / (tpOut + fnOut)
    else:
        TPrateOut = nan

    # Positive predictive value (precision)
    if tpOut + fpOut > 0:
        PPVrateOut = tpOut / (tpOut + fpOut)
    else:
        PPVrateOut = nan

    # F1 rate
    if TPrateOut + PPVrateOut > 0:
        F1Out = 2 * (TPrateOut * PPVrateOut) / (TPrateOut + PPVrateOut)
    else:
        F1Out = nan

    return tpIn, tnIn, fpIn, fnIn, \
           tpOut, tnOut, fpOut, fnOut, \
           TPrateIn, TPrateOut, TPrateIn * probIn + TPrateOut * (1 - probIn), \
           PPVrateIn, PPVrateOut, PPVrateIn * probIn + PPVrateOut * (1 - probIn), \
           F1In, F1Out, F1In * probIn + F1Out * (1 - probIn)
