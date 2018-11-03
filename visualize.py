import os
import numpy as np
import matplotlib.pyplot as plt

results_dir = os.getcwd() + "/results/"
figure_dir = os.getcwd() + "/pics/"


def smooth(time_series, degree):
    """
    Input I: Time series and smoothing degree
    Output: Computes moving average for a time series (rets/steps) and order n
    """
    ret = np.cumsum(time_series, dtype=float)
    ret[degree:] = ret[degree:] - ret[:-degree]
    return ret / degree


def plot_sequence(time, sequence, hidden, save_pic=False):
    plt.scatter(time, sequence, s=0.5, c=hidden)
    plt.title("HHMM-Sampled Stimulus Sequence for {} Timesteps".format(int(max(time)) + 1))
    if save_pic:
        plt.savefig(figure_dir + "hhmm_seq.png", dpi=300)
    else:
        plt.show()


def plot_surprise(time, surprisals, title, degree=1, save_pic=False):

    colours = ["r", "b", "g"]
    labels = ["Predictive Surprise",
              "Bayesian Surprise",
              "Confidence-Corrected Surprise"]

    for i, surprisal in enumerate(surprisals):
        surprisal = smooth(surprisal, degree)
        plt.plot(time, surprisal, c=colours[i], label=labels[i])

    plt.title(title)
    plt.legend()

    if save_pic:
        plt.savefig(figure_dir + "surprise.png", dpi=300)
    else:
        plt.show()
    return


def plot_beta_evolution(time, alpha, beta, save_pic=False):
    if save_pic:
        plt.savefig(figure_dir + "distribution.png", dpi=300)
    else:
        plt.show()
    return

if __name__ == "__main__":
    sbl_surprise = np.loadtxt(results_dir + "sbl_surprise_TP_200.txt")

    # Unpack array for easier reading
    time = sbl_surprise[:, 0]
    sequence = sbl_surprise[:, 1]
    hidden = sbl_surprise[:, 2]

    predictive_surprisal = sbl_surprise[:, 3]
    bayesian_surprisal = sbl_surprise[:, 4]
    corrected_surprisal = sbl_surprise[:, 5]

    if sbl_surprise.shape < 9:
        alpha = sbl_surprise[:, 6]
        beta = sbl_surprise[:, 7]
    else:
        alpha_0 = sbl_surprise[:, 6]
        alpha_1 = sbl_surprise[:, 7]
        beta_0 = sbl_surprise[:, 8]
        beta_1 = sbl_surprise[:, 9]

    # Input Array: [t, obs_t, hidden_t, PS_t, BS_t, CS_t] + [alphas, betas]
    plot_sequence(time, sequence, hidden, save_pic=False)
    plot_surprise(time, [predictive_surprisal, bayesian_surprisal, corrected_surprisal], "Surprisal Comparison", save_pic=False)
