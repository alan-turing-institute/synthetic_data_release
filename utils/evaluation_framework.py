"""
Procedures for running a privacy evaluation on a generative model
"""

from numpy import where, mean, nan

from utils.constants import *


def get_accuracy(guesses, labels, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    pIn = sum([g == l for g,l in zip(guesses[idxIn], labels[idxIn])])/len(idxIn)
    pOut = sum([g == l for g,l in zip(guesses[idxOut], labels[idxOut])])/len(idxOut)

    return pIn, pOut


def get_tp_fp_rates(guesses, labels):
    targetIn = where(labels == LABEL_IN)[0]
    targetOut = where(labels == LABEL_OUT)[0]
    return sum(guesses[targetIn] == LABEL_IN)/len(targetIn), sum(guesses[targetOut] == LABEL_IN)/len(targetOut)


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
    return 1.0 - sum(after[idxIn]/len(idxIn))


def get_prob_success_total(pCorrectIn, pCorrectOut, probIn):
    """Calculate total probability of success by weighing the
    two scenarios of including or not including the target record
    in the training data."""
    return pCorrectIn * probIn + pCorrectOut * (1 - probIn)


def get_TP_rates(guesses, labels, targetPresence, positive_label, probIn):
    """Calculate true and total positive and true positive rates for classification task"""
    idxIn = where(targetPresence == LABEL_IN)[0]
    guessesIn = guesses[idxIn]
    labelsIn = labels[idxIn]
    labelsInPos = where(labelsIn == positive_label)[0]
    TruePositivesIn = sum([g == l for g, l in zip(guessesIn[labelsInPos], labelsIn[labelsInPos])])
    PositivesIn = len(labelsInPos)
    if PositivesIn > 0:
        TPrateIn = TruePositivesIn / PositivesIn
    else:
        TPrateIn = nan

    idxOut = where(targetPresence == LABEL_OUT)[0]
    guessesOut = guesses[idxOut]
    labelsOut = labels[idxOut]
    labelsOutPos = where(labelsOut == positive_label)[0]
    TruePositivesOut = sum([g == l for g, l in zip(guessesOut[labelsOutPos], labelsOut[labelsOutPos])])
    PositivesOut = len(labelsOutPos)
    if PositivesOut > 0:
        TPrateOut = TruePositivesOut / PositivesOut
    else:
        TPrateOut = nan

    return TruePositivesIn, PositivesIn, TruePositivesOut, PositivesOut, \
           TPrateIn, TPrateOut, TPrateIn * probIn + TPrateOut * (1 - probIn)
