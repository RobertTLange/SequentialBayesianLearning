from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
from seq_gen import seq_gen

import dit
from dit.divergences import kullback_leibler_divergence, jensen_shannon_divergence, cross_entropy

results_dir = os.getcwd() + "/results/"
fig_dir = os.getcwd() + "/pics/"

order = 2
prob_regime_init = np.array([0.5, 0.5])
prob_regime_change = 0.01
prob_obs_init = np.array([0.5, 0.5, 0])
prob_catch = 0.05

seq_length = 100000


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

    for t in range(1, deviants.shape[0]):
        if sequence[t, 2] != sequence[t-1, 2]:
            deviants[t, 0] = sequence[t, 2]
            deviants[t, 1] = sequence[t, 1]
            deviants[t, 2] = 1
            deviants[t, 3] = counter
            counter = 0
        else:
            counter += 1
    return deviants


def empirical_stand_distr(sequence):
    """
    INPUT:
        * sequence - Sequence sampled from seq_gen class
    OUTPUT:
        * js_temp - Jensen-Shannon-Divergence between standard-between-dev
                    empirical distributions compared between regimes
    """
    deviants = find_deviants(sequence)
    reg_0 = deviants[deviants[:, 1] == 0, :]
    reg_1 = deviants[deviants[:, 1] == 1, :]

    epmf_reg_0 = np.histogram(reg_0[:, 3], bins=int(np.max(reg_0[:, 3])),
                              density=True)
    epmf_reg_1 = np.histogram(reg_1[:, 3], bins=int(np.max(reg_1[:, 3])),
                              density=True)

    d1 = dit.ScalarDistribution(epmf_reg_0[1][:-1], epmf_reg_0[0])
    d2 = dit.ScalarDistribution(epmf_reg_1[1][:-1], epmf_reg_1[0])
    js_temp =jensen_shannon_divergence([d1, d2])
    return js_temp, reg_0, reg_1

def main():
    empirical_stand_distr(sequence)
    return

def plot_all(reg_0, reg_1, seq_gen_temp):
    plt.hist(reg_0[:, 3], density=True, label="Regime 0")
    plt.hist(reg_1[:, 3], density=True, label="Regime 1")
    # plt.title(r"$p_{0|00}^{(0)}={}, p_{0|01}^{(0)}=0.{}, p_{0|10}^{(0)}=0.{},  p_{0|11}^{(0)}={}, p_{0|00}^{(1)}={}, p_{0|01}^{(1)}={}, p_{0|10}^{(1)}={},  p_{0|11}^{(1)}={}$".format(seq_gen_temp.prob_obs_change))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # main()
    prob_obs_change = [0.35, 0.35, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35]

    seq_gen_temp = seq_gen(order, prob_catch, prob_regime_init,
                           prob_regime_change, prob_obs_init, prob_obs_change)
    sequence = seq_gen_temp.sample(seq_length)

    js_temp, reg_0, reg_1 = empirical_stand_distr(sequence)
    plot_all(reg_0, reg_1, seq_gen_temp)
