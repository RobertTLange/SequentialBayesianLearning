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
    deviants = np.zeros((sequence.shape[0], 3))
    counter = 1

    regime_switches = 0

    for t in range(1, deviants.shape[0]):
        if sequence[t, 2] != sequence[t-1, 2]:
            # Check for alternation
            deviants[t, 0] = sequence[t, 1]  # hidden state
            deviants[t, 1] = 1 # change indicator
            deviants[t, 2] = counter
            counter = 0
        else:
            counter += 1

        if sequence[t, 1] != 2 and sequence[t-1, 1] != 2:
            if sequence[t, 1] != sequence[t-1, 1]:
                regime_switches += 1

    dev_out = deviants[deviants[:, 1] == 1, :]
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

    # 0th Order Stimulus probability (empirical)
    stim_prob_reg0 = np.mean(sequence[sequence[:, 1] == 0, 2])
    stim_prob_reg1 = np.mean(sequence[sequence[:, 1] == 1, 2])

    # Empirical pmf of standards between deviants for both regimes
    reg_0_dev = deviants[deviants[:, 0] == 0, :]
    reg_1_dev = deviants[deviants[:, 0] == 1, :]

    epmf_reg_0_dev = np.histogram(reg_0_dev[:, 2],
                                  bins=int(np.max(reg_0_dev[:, 2])),
                                  density=True)
    epmf_reg_1_dev = np.histogram(reg_1_dev[:, 2],
                                  bins=int(np.max(reg_1_dev[:, 2])),
                                  density=True)

    # Calculate symmetric Jensen - Shannon divergence
    d1 = dit.ScalarDistribution(epmf_reg_0_dev[1][:-1], epmf_reg_0_dev[0])
    d2 = dit.ScalarDistribution(epmf_reg_1_dev[1][:-1], epmf_reg_1_dev[0])
    js_temp = jensen_shannon_divergence([d1, d2])

    if verbose:
        print("Empirical Probabilities: \n Empirical Catch Prob.: {} \n Empirical Regime Switch Prob.: {} \n Empirical Overall High-Intensity Stimulus Prob.: {} \n Empirical Regime 0 High-Intensity Stimulus Prob.: {} \n Empirical Regime 1 High-Intensity Stimulus Prob.: {} \n JS Div. Deviant Waiting Time Distr. between Regimes: {}".format(catch_prob, switch_prob, stim_prob_overall, stim_prob_reg0, stim_prob_reg1, js_temp))
        print("--------------------------------------------")

    stats_out = {"emp_catch_prob": catch_prob,
                 "emp_overall_sp": stim_prob_overall,
                 "emp_reg0_sp": stim_prob_reg0,
                 "emp_reg1_sp": stim_prob_reg1,
                 "js_div": js_temp}

    return stats_out, reg_0_dev, reg_1_dev


def main(order, verbose, plot, save):
    prob_regime_init = np.array([0.5, 0.5])
    prob_obs_init = np.array([0.5, 0.5, 0])

    prob_regime_change = 0.01
    prob_catch = 0.05

    seq_length = 100000

    gen_models = []
    stats = []
    reg_0s = []
    reg_1s = []

    for i in range(1, 10):
        if order == 1:
            prob = [i/2*0.1, 1-i/2*0.1, 1-i/2*0.1, i/2*0.1]
        elif order == 2:
            prob = [i/2*0.1, i/2*0.1, 1-i/2*0.1, 1-i/2*0.1,
                    1-i/2*0.1, 1-i/2*0.1, i/2*0.1, i/2*0.1]
        seq_gen_temp = seq_gen(order, prob_catch, prob_regime_init,
                               prob_regime_change, prob_obs_init, prob,
                               verbose=False)
        sequence = seq_gen_temp.sample(seq_length)
        stats_temp, reg_0_dev, reg_1_dev = calc_stats(sequence, verbose)

        gen_models.append(seq_gen_temp)
        stats.append(stats_temp)
        reg_0s.append(reg_0_dev)
        reg_1s.append(reg_1_dev)

    if plot:
        plot_all(reg_0s, reg_1s, gen_models, stats, order, save)
    return


def plot_all(reg_0s, reg_1s, gen_models, stats, order, save):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    fig.tight_layout()

    counter = 0
    for i in range(3):
        for j in range(3):
            ax[i, j].hist(reg_0s[counter][:, 2], density=True, label="Regime 0", alpha=0.5, range=(0, 20))
            ax[i, j].hist(reg_1s[counter][:, 2], density=True, label="Regime 1", alpha=0.5, range=(0, 20))

            if order == 1:
                ax[i, j].set_title("R0: p(0|0)={}, p(0|1)={}; R1: p(0|0)={}, p(0|1)={}".format(*gen_models[counter].prob_obs_change), fontsize=8)
            elif order == 2:
                ax[i, j].set_title(" Prob. Regime 0: p(0|00)={}, p(0|01)={}, p(0|10)={},  p(0|11)={} \n Prob. Regime 1: p(0|00)={}, p(0|01)={}, p(0|10)={},  p(0|11)={}".format(*gen_models[counter].prob_obs_change), fontsize=4)

            # Add extra info as additional lines with label in legend
            ax[i, j].plot([], [], ' ', label="Emp. R0 High-I SP: {}".format(round(stats[counter]["emp_reg0_sp"], 3)))
            ax[i, j].plot([], [], ' ', label="Emp. R1 High-I SP: {}".format(round(stats[counter]["emp_reg1_sp"], 3)))
            ax[i, j].plot([], [], ' ', label="JS-Div. Deviants: {}".format(round(stats[counter]["js_div"], 3)))
            counter += 1
            ax[i,j].legend()

            if i != 2:
                ax[i,j].set_xticks(())

            if j != 0:
                ax[i,j].set_yticks(())

            if i == 2:
                ax[i,j].set_xlabel("Emp. Distr. - Standards between Deviants")

    if save:
        print(1)
        plt.savefig("data_gen_comparison_" + str(order) + ".png", dpi=300)
    else:
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
    parser.add_argument('-p', '--plot_stats',
                        action="store_true",
                        default=False,
						help='View/Plot the sampled sequence')
    parser.add_argument('-s', '--save',
                        action="store_true",
                        default=False,
                        help='Save the generated figure')

    args = parser.parse_args()
    verbose = args.verbose
    order = args.markov_order
    plot = args.plot_stats
    save = args.save

    main(order, verbose, plot, save)

    """
    python seq_analysis.py -p -order 1 -v
    python seq_analysis.py -p -order 2 -v
    """
