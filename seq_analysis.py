from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import argparse
from seq_gen import *

import dit
from dit.divergences import jensen_shannon_divergence

results_dir = os.getcwd() + "/results/"
fig_dir = os.getcwd() + "/pics/"

np.random.seed(222)

# sequence_meta = {"sample_output": sequence,
#                  "prob_regime_init": seq_gen_temp.prob_regime_init,
#                  "prob_obs_init": seq_gen_temp.prob_obs_init,
#                  "prob_obs_change": seq_gen_temp.prob_obs_change,
#                  "prob_regime_change": seq_gen_temp.prob_regime_change}

def find_deviants(sequence):
    """
    INPUT:
        * sequence - Sequence sampled from seq_gen class
    OUTPUT:
        * deviants
            - 1st col: Sequence
            - 2nd col: Hidden state
            - 3rd col : 1 - stim is deviant, 0 - stim is standard
            - 4th col: time since last deviants/number of standards before
    """
    deviants = np.zeros((sequence.shape[0], 4))
    counter = 1

    regime_switches = 0

    for t in range(1, deviants.shape[0]):
        if sequence[t, 2] != sequence[t-1, 2]:
            deviants[t, 0] = sequence[t, 2]
            deviants[t, 1] = sequence[t, 1]
            deviants[t, 2] = 1
            deviants[t, 3] = counter
            counter = 0
        else:
            counter += 1

        if sequence[t, 1] != 2 and sequence[t-1, 1] != 2:
            if sequence[t, 1] != sequence[t-1, 1]:
                regime_switches += 1
    return deviants, regime_switches


def calc_stats(sequence, verbose):
    """
    INPUT:
        * sequence - Sequence sampled from seq_gen class
    OUTPUT:
        * js_temp - Jensen-Shannon-Divergence between standard-between-dev
                    empirical distributions compared between regimes
    """

    sequence_sub = sequence[sequence[:, 2] != 0.5, :]
    deviants, regime_switches = find_deviants(sequence)

    # Catch trial/regime switch prob
    catch_prob = len(sequence[sequence[:, 2] == 0.5, 0])/sequence.shape[0]
    switch_prob = regime_switches/sequence.shape[0]

    stim_prob_overall = len(sequence[sequence[:, 2] == 1, 2])/(len(sequence[sequence[:, 2] == 1, 2]) + len(sequence[sequence[:, 2] == 0, 2]))

    # Filter stimuli based on regime


    # 0th Order Stimulus probability (empirical)
    stim_prob_reg0 = np.mean(sequence[sequence[:, 1] == 0, 2])
    stim_prob_reg1 = np.mean(sequence[sequence[:, 1] == 1, 2])

    if verbose:
        print("Empirical Probabilities: \n Empirical Catch Prob.: {} \n Empirical Regime Switch Prob.: {} \n Empirical Overall High-Intensity Stimulus Prob.: {} \n Empirical Regime 0 High-Intensity Stimulus Prob.: {} \n Empirical Regime 1 High-Intensity Stimulus Prob.: {}".format(catch_prob, switch_prob, stim_prob_overall, stim_prob_reg0, stim_prob_reg1))
        print("--------------------------------------------")
    # 1st Order Transition probability (empirical)

    # 2nd Order Transition probability (empirical)

    # Empirical pmf of standards between deviants for both regimes
    try:
        epmf_reg_0 = np.histogram(reg_0[:, 3], bins=int(np.max(reg_0[:, 3])),
                                  density=True)
        epmf_reg_1 = np.histogram(reg_1[:, 3], bins=int(np.max(reg_1[:, 3])),
                                  density=True)
        # Calculate symmetric Jensen - Shannon divergence
        d1 = dit.ScalarDistribution(epmf_reg_0[1][:-1], epmf_reg_0[0])
        d2 = dit.ScalarDistribution(epmf_reg_1[1][:-1], epmf_reg_1[0])
        js_temp = jensen_shannon_divergence([d1, d2])
    except:
        js_temp = None
    return js_temp

def main(order, verbose):
    order = 2
    prob_regime_init = np.array([0.5, 0.5])
    prob_obs_init = np.array([0.5, 0.5, 0])

    prob_regime_change = 0.01
    prob_catch = 0.05

    seq_length = 100000

    for i in range(1, 10):
        prob = np.repeat(i*0.1, 2*2**order)
        seq_gen_temp = seq_gen(order, prob_catch, prob_regime_init,
                               prob_regime_change, prob_obs_init, prob,
                               verbose=False)
        sequence = seq_gen_temp.sample(seq_length)
        js_temp, reg_0_dev, reg_1_dev = calc_stats(sequence, verbose)

    # plot_all(reg_0, reg_1, seq_gen_temp)
    return


def plot_all(reg_0, reg_1, seq_gen_temp):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    fig.tight_layout()

    for i in range(3):
        for j in range(3):
            ax[i, j].hist(reg_0[:, 3], density=True, bins=np.arange(20).tolist(), label="Regime 0", alpha=0.5)
            ax[i, j].hist(reg_1[:, 3], density=True, bins=np.arange(20).tolist(), label="Regime 1", alpha=0.5)
            ax[i, j].set_title(" Prob. Regime 0: p(0|00)={}, p(0|01)={}, p(0|10)={},  p(0|11)={} \n Prob. Regime 1: p(0|00)={}, p(0|01)={}, p(0|10)={},  p(0|11)={}".format(*seq_gen_temp.prob_obs_change), fontsize=6)
            ax[i, j].legend()
    plt.show()


def draw_dirichlet_params(alphas):
    if len(alphas) != 8:
        raise ValueError("Provide correct size of concentration params")
    return np.random.dirichlet((alphas), 1).transpose()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-order', '--markov_order', action="store",
                        default=1, type=int,
						help='Markov dependency on observation level')
    parser.add_argument('-v', '--verbose',
                        action="store_true",
                        default=False,
						help='Get status printed out')

    args = parser.parse_args()
    verbose = args.verbose
    order = args.markov_order

    main(order, verbose)
