import os
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import normalize

fig_dir = os.getcwd() + "/figures/"


def preproc_surprisal(SP, AP, TP):
    time = SP["time"]
    hidden = SP["hidden"]
    sequence = SP["sequence"]

    PS = [normalize(SP["predictive_surprise"]),
          normalize(AP["predictive_surprise"]),
          normalize(TP["predictive_surprise"])]
    BS = [normalize(SP["bayesian_surprise"]),
          normalize(AP["bayesian_surprise"]),
          normalize(TP["bayesian_surprise"])]
    CS = [normalize(SP["confidence_corrected_surprise"]),
          normalize(AP["confidence_corrected_surprise"]),
          normalize(TP["confidence_corrected_surprise"])]

    return time, hidden, sequence, PS, BS, CS


def plot_surprise(SP, AP, TP, title="Categorical-Dirichlet",
                  save_pic=False):

    time, hidden, sequence, PS, BS, CS = preproc_surprisal(SP, AP, TP)

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 8))
    fig.suptitle('SBL {} Agent'.format(title), fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    ax[0].set_xlim([0, max(time)])
    ax[0].set_ylim([-0.2, 1.2])
    ax[0].set_title(r"Hidden State Sequence: $s_1, \dots, s_t$", fontsize=10)
    ax[0].scatter(time, hidden, s=0.5)
    ax[0].set_xticks([], [])

    ax[1].set_xlim([0, max(time)])
    ax[1].set_ylim([-0.2, 1.2])
    ax[1].set_title(r"Observation Sequence: $o_1, \dots, o_t$", fontsize=10)
    ax[1].scatter(time, sequence, s=0.5)
    ax[1].set_xticks([], [])

    # Plot initial layout that persists (isn't redrawn)
    ## First row - Sequence of Trials
    sub_t = ["Stimulus Probability", "Alternation Probability", "Transition Probability"]
    for i in range(2, 5):
        ax[i].set_xlim([0, max(time)])
        ax[i].set_ylim([-0.2, 1.2])
        if i == 2:
            ax[i].set_ylim([-0.2, 1.5])
        ax[i].plot(time, PS[i-2], c="r", label=r"Predictive Surprise: $PS(o_t)$")
        ax[i].plot(time, BS[i-2], c="b", label=r"Bayesian Surprise: $BS(o_t)$")
        ax[i].plot(time, CS[i-2], c="g", label=r"Confidence-Corrected Surprise: $CS(o_t)$")
        ax[i].set_title("{} Model".format(sub_t[i-2]), fontsize=10)
        if i==2:
            ax[i].legend(loc="upper center", prop={'size': 6}, ncol=3)
        if i!=4:
            ax[i].set_xticks([], [])
    if save_pic:
        plt.savefig(figure_dir + 'sbl_cd_comparison.png', dpi=300)
    else:
        plt.show()
