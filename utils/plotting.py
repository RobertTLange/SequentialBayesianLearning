import os
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats

fig_dir = os.getcwd() + "/figures/"

# Define color blind-friendly color cycle
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def smooth(ts, windowSize):
    # Perform smoothed moving average with specified window to time series
    weights = np.repeat(1.0, windowSize) / windowSize
    ts_MA = np.convolve(ts, weights, 'valid')
    return ts_MA


def plot_sequence(sequence, hidden, max_t, save_fname=None):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

    sequence = sequence.astype(float)
    hidden = hidden.astype(float)

    idx = np.argwhere(sequence == 2)
    sequence[idx] = 0.5
    hidden[idx] = 0.5
    time = np.arange(max_t)

    ax[0].set_xlim([0, max_t])
    ax[0].set_ylim([-0.2, 1.2])
    ax[0].set_title(r"Hidden State Sequence: $s_1, \dots, s_t$", fontsize=15)
    ax[0].scatter(time[:max_t], hidden[:max_t], s=10)
    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])

    ax[1].set_xlim([0, max_t])
    ax[1].set_ylim([-0.2, 1.2])
    ax[1].set_title(r"Observation Sequence: $o_1, \dots, o_t$", fontsize=15)
    ax[1].scatter(time[:max_t], sequence[:max_t], s=10)
    ax[1].set_yticks([], [])
    ax[1].set_xlabel(r"Trial $t$", fontsize=15)

    fig.tight_layout()
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300)
    else:
        plt.show()


def preproc_surprisal(SP, AP, TP, norm):
    time = SP["time"]
    hidden = SP["hidden"]
    sequence = SP["sequence"]
    sequence[sequence == 2] = 0.5
    hidden[hidden == 2] = 0.5

    catch_id = np.argwhere(sequence == 0.5)

    if norm:
        PS = [normalize(SP["predictive_surprise"], catch_id),
              normalize(AP["predictive_surprise"], catch_id),
              normalize(TP["predictive_surprise"], catch_id)]
        BS = [normalize(SP["bayesian_surprise"], catch_id),
              normalize(AP["bayesian_surprise"], catch_id),
              normalize(TP["bayesian_surprise"], catch_id)]
        CS = [normalize(SP["confidence_corrected_surprise"], catch_id),
              normalize(AP["confidence_corrected_surprise"], catch_id),
              normalize(TP["confidence_corrected_surprise"], catch_id)]
    else:
        PS = [SP["predictive_surprise"],
              AP["predictive_surprise"],
              TP["predictive_surprise"]]
        BS = [SP["bayesian_surprise"],
              AP["bayesian_surprise"],
              TP["bayesian_surprise"]]
        CS = [SP["confidence_corrected_surprise"],
              AP["confidence_corrected_surprise"],
              TP["confidence_corrected_surprise"]]
    return time, hidden, sequence, PS, BS, CS


def plot_surprise2(surprise, preproc="normalize",
                   sub_t="Stimulus Probablity Model",
                   title="Categorical-Dirichlet",
                   max_t=800, save_fname=None):

    time = surprise["time"]
    hidden = surprise["hidden"]
    sequence = surprise["sequence"]
    sequence[sequence == 2] = 0.5
    hidden[hidden == 2] = 0.5

    PS =  stats.zscore(surprise["predictive_surprise"])
    BS =  stats.zscore(surprise["bayesian_surprise"])
    CS =  stats.zscore(surprise["confidence_corrected_surprise"])

    hidden[hidden == 1] = np.max(BS[:max_t]) + 0.3
    hidden[hidden == 0] = np.min(PS[:max_t]) - 0.3
    sequence[sequence == 1] = np.max(BS[:max_t]) + 0.9
    sequence[sequence == 0] = np.min(PS[:max_t]) - 0.9

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('SBL {} Agent'.format(title), fontsize=16)

    # Plot the observed trial sequence and hidden states
    ax.scatter(time[:max_t], hidden[:max_t], s=4, label=r"Regime: $s_t$", c="#16738F")
    ax.scatter(time[:max_t], sequence[:max_t], s=4, label=r"Stimulus: $o_t$", c="#B31329")

    ax.set_xlim([0, max_t])
    ax.set_ylim([np.min(PS[:max_t]) -1.2, np.max(BS[:max_t]) + 1.2])
    # ax.set_yticks([0, 0.5, 1])
    ax.plot(time[:max_t], PS[:max_t], c="r",
            label=r"Predictive Surprise: $PS(y_t)$")
    ax.plot(time[:max_t], BS[:max_t], c="b",
            label=r"Bayesian Surprise: $BS(y_t)$")
    ax.plot(time[:max_t], CS[:max_t], c="g",
            label=r"Confidence-Corrected Surprise: $CS(y_t)$")

    ax.set_title(sub_t, fontsize=15)
    ax.legend(loc="center right", prop={'size': 12}, ncol=1)

    ax.set_xlabel("Trial (t)", fontsize=15)
    ax.set_ylabel("Surprise Regressors", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fname is not None:
        plt.savefig(save_fname, dpi=600)
        print("Saved figure to {}".format(save_fname))
    else:
        plt.show()


def plot_surprise(SP, AP, TP, normalize=True,
                  title="Categorical-Dirichlet",
                  max_t=800, save_fname=None):

    time, hidden, sequence, PS, BS, CS = preproc_surprisal(SP, AP, TP,
                                                           normalize)

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(12, 12))
    fig.suptitle('SBL {} Agent'.format(title), fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    ax[0].set_xlim([0, max_t])
    ax[0].set_ylim([-0.2, 1.2])
    ax[0].set_title(r"Hidden State Sequence: $s_1, \dots, s_t$", fontsize=15)
    ax[0].scatter(time[:max_t], hidden[:max_t], s=4)
    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])

    ax[1].set_xlim([0, max_t])
    ax[1].set_ylim([-0.2, 1.2])
    ax[1].set_title(r"Observation Sequence: $o_1, \dots, o_t$", fontsize=15)
    ax[1].scatter(time[:max_t], sequence[:max_t], s=4)
    ax[1].set_xticks([], [])
    ax[1].set_yticks([], [])

    # Plot initial layout that persists (isn't redrawn)
    # First row - Sequence of Trials
    sub_t = ["Stimulus Probability", "Alternation Probability",
             "Transition Probability"]
    for i in range(2, 5):
        ax[i].set_xlim([0, max_t])
        ax[i].set_ylim([-0.2, np.max(PS[i-2][:max_t] + 0.2)])
        if i == 2:
            ax[i].set_ylim([-0.2, np.max(PS[i-2][:max_t] + 0.2)])
        ax[i].plot(time[:max_t], PS[i-2][:max_t], c="r",
                   label=r"Predictive Surprise: $PS(o_t)$")
        ax[i].plot(time[:max_t], BS[i-2][:max_t], c="b",
                   label=r"Bayesian Surprise: $BS(o_t)$")
        # ax[i].plot(time[:max_t], CS[i-2][:max_t], c="g",
        #            label=r"Confidence-Corrected Surprise: $CS(o_t)$")
        ax[i].set_title("{} Model".format(sub_t[i-2]), fontsize=15)
        if i == 2:
            ax[i].legend(loc="upper right", prop={'size': 12}, ncol=1)
        if i != 4:
            ax[i].set_xticks([], [])
    if save_fname is not None:
        plt.savefig(save_fname, dpi=600)
        print("Saved figure to {}".format(save_fname))
    else:
        plt.show()


def plot_free_energy(fe_ts_list, windowSize=5,
                     labels=["Null-Model",
                             "Robust Ridge Regr.",
                             "Hierarchical GLM",
                             "B-NN (10 Hiddens)"],
                     save_fname=None):
    """
    Plot the time-series of the optimization of the free energy/ELBO
    """
    plt.figure(figsize=(8, 8))
    for i in range(len(fe_ts_list)):
        plt.plot(smooth(fe_ts_list[i], windowSize), label=labels[i])

    plt.xlabel("VI Optimization Updates")
    plt.ylabel("LME/Negative Free Energy")
    plt.legend()
    plt.title("Free Energy/ELBO after ADVI Optimization")

    if save_fname is not None:
        plt.savefig(fig_dir + save_fname, dpi=300)
    else:
        plt.show()


def plot_lme_across_int(y_tw, null_model_lme, reg_model_lme,
                        reg_label, save_fname=None):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.title("LME across Interstimulus Interval (Normalized by Null)")
    plt.plot(y_tw, reg_model_lme-null_model_lme)
    plt.ylabel("LME/Neg. FE")
    plt.xticks([])

    plt.subplot(2, 1, 2)
    plt.plot(y_tw, reg_model_lme, label=reg_label)
    plt.plot(y_tw, null_model_lme, label="Null")
    plt.xlabel("Inter-Stimulus Interval")
    plt.ylabel("LME/Neg. FE")
    plt.legend(loc=1)

    if save_fname is not None:
        plt.savefig(fig_dir + save_fname, dpi=300)
    else:
        plt.show()


def heatmap_lme(lme_array, x_labels, y_labels,
                title="Log Model Evidences: Cz"):
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(lme_array, cmap="jet")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(np.around(x_labels, 2))
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)
    ax.set_title(title)
    ax.set_xlabel("Inter-Stimulus Time Interval")
    ax.set_ylabel("Regressor Model")
