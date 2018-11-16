import os
import pickle
import numpy as np
import scipy.stats as ss
from scipy.special import gamma, digamma

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.style.use('ggplot')

results_dir = os.getcwd() + "/results/"
figure_dir = os.getcwd() + "/pics/"


def load_obj(title):
    with open(title, 'rb') as f:
        return pickle.load(f)

def plot_sequences(times, sequences, hiddens, subtitles_seq, plot_title,
                   save_pic=False, file_title="temp.png"):

    fig, ax = plt.subplots(nrows=len(sequences), ncols=1, figsize=(18, 12))
    fig.suptitle(plot_title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    for i in range(len(sequences)):
        coloring = []
        for j in range(len(hiddens[i])):
            if hiddens[i][j] == 0:
                coloring += "g"
            elif hiddens[i][j] == 1:
                coloring += "b"
            else:
                coloring += "r"

        ax[i].scatter(times[i], sequences[i], s=4.5, c=coloring)
        ax[i].set_title(subtitles_seq[i])
    if save_pic:
        plt.savefig(figure_dir + file_title, dpi=300)
    else:
        plt.show()


def plot_surprise(time, surprisals, title, save_pic=False):

    colours = ["r", "b", "g"]
    labels = ["Predictive Surprise",
              "Bayesian Surprise",
              "Confidence-Corrected Surprise"]

    for i, surprisal in enumerate(surprisals):
        plt.plot(time, surprisal, c=colours[i], label=labels[i])

    plt.title(title)
    plt.legend()

    if save_pic:
        plt.savefig(figure_dir + "surprise.png", dpi=300)
    else:
        plt.show()
    return


def calc_all_KLs(alpha, beta):
    T = len(alpha)
    KL = np.zeros((len(alpha), 1))

    for i in range(1, T):
        KL[i] = calculate_KL_beta(alpha[i-1], alpha[i],
                                  beta[i-1], beta[i])
    return KL


def calculate_KL_beta(alpha_0, alpha_1, beta_0, beta_1):
    return np.log(gamma(alpha_0)) - np.log(gamma(alpha_1)) + np.log(gamma(beta_1)) + (alpha_1 - beta_1)*(digamma(alpha_1) - digamma(alpha_0))


def draw_animation(time, sequence, hidden,
                   surprisal, alpha, beta, type="Alternation Probabiltiy: Beta-Bernoulli",
                   save_pic=False):

    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.suptitle('SBL - {} Agent'.format(type), fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.925])

    # Plot initial layout that persists (isn't redrawn)
    ## First row - Sequence of Trials
    ax[0].set_xlim([0, max(time)])
    ax[0].set_ylim([-0.2, 1.2])
    ax[0].set_title(r"Observed Sequence of Trials: $o_1, \dots, o_t$", fontsize=10)
    ax[0].scatter(-1,-1,c="darkslateblue", s=0.5, label=r"Slow Alternation Regime")
    ax[0].scatter(-1,-1,c="y", s=0.5,  label=r"Fast Alternation Regime")
    ax[0].legend(loc=7, prop={'size': 6})

    ## Second row - Surprisals
    ax[1].set_xlim([0, max(time)])
    ax[1].set_ylim([-0.2, 5])
    ax[1].plot(0,0,c="r", label=r"Predictive Surprise: $PS(o_t)$")
    ax[1].plot(0,0,c="b", label=r"Bayesian Surprise: $BS(o_t)$")
    ax[1].plot(0,0,c="g", label=r"Confidence-Corrected Surprise: $CS(o_t)$")
    ax[1].set_title("Suprisal of SBL Agent", fontsize=10)
    ax[1].legend(loc=1, prop={'size': 6})

    ## Change in distributions KL(posterior_t||posterior_t-1)
    ax[2].set_xlim([0, 1])
    ax[2].set_ylim([-0.2, 7])
    line, = ax[2].plot(np.linspace(0, 1, 5000), np.repeat(0, 5000) ,color='red', lw=2, ls='-', alpha=0.5)
    ax[2].set_title(r"Beta Posterior of Hidden State: $p(s_t|o_{1:t})$", fontsize=10)

    def update(t, sequence=sequence, hidden=hidden, suprisal=surprisal,alpha=alpha, beta=beta):
        labe = 'Timestep {0}'.format(t)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.

        ax[2].set_xlabel(labe)

        ax[0].scatter(time[:t], sequence[:t], s=0.5, c=hidden[:t])
        colours = ["r", "b", "g"]

        for i in range(surprisal.shape[1]):
            surprisal_temp = surprisal[:t, i]
            ax[1].plot(time[:t], surprisal_temp, c=colours[i])

        x = np.linspace(0, 1, 5000)
        y = ss.beta.pdf(x, alpha[t], beta[t])
        line.set_ydata(y)
        return ax

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, int(max(time)), 5), interval=400)
    if save_pic:
        anim.save(figure_dir + 'sbl_bb.gif', dpi=300, writer='imagemagick')
    else:
        plt.show()


def plot_final(time, sequence, hidden,
               surprisal, alpha, beta, type="Beta-Bernoulli",
               save_pic=False):
    fig, ax = plt.subplots(nrows=4, ncols=1)
    fig.suptitle('SBL {} Agent'.format(type), fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.925])

    ax[0].set_xlim([0, max(time)])
    ax[0].set_ylim([-0.2, 1.2])
    ax[0].set_title(r"Observed Sequence of Trials: $o_1, \dots, o_t$", fontsize=10)
    ax[0].scatter(time, sequence, s=0.5, c=hidden)
    ax[0].scatter(-1,-1,c="darkslateblue",s=0.5,label=r"Slow Alternation Regime")
    ax[0].scatter(-1,-1,c="y",s=0.5,label=r"Fast Alternation Regime")
    ax[0].legend(loc=7, prop={'size': 6})

    # Plot initial layout that persists (isn't redrawn)
    ## First row - Sequence of Trials
    sub_t = ["Stimulus Probability", "Alternation Probability", "Transition Probability"]
    for i in range(1, 4):
        ax[i].set_xlim([0, max(time)])
        ax[i].set_ylim([-0.2, 1.2])
        ax[i].plot(time,surprisal[i-1][:, 0],c="r", label=r"Predictive Surprise: $PS(o_t)$")
        ax[i].plot(time,surprisal[i-1][:, 1],c="b", label=r"Bayesian Surprise: $BS(o_t)$")
        ax[i].plot(time,surprisal[i-1][:, 2],c="g", label=r"Confidence-Corrected Surprise: $CS(o_t)$")
        ax[i].set_title("{} Model".format(sub_t[i-1]), fontsize=10)
        if i==1:
            ax[i].legend(loc="upper center", prop={'size': 6}, ncol=3)

    if save_pic:
        plt.savefig(figure_dir + 'sbl_bb_comparison.png', dpi=300)
    else:
        plt.show()

def standardize(surprise):
    arr = np.array(surprise)
    temp = arr/np.nanmax(arr,axis=0)
    return temp


if __name__ == "__main__":

    files_1st_order = ["1st_5_01_5_10_200.pkl",
                       "1st_5_01_5_25_200.pkl",
                       "1st_5_01_5_40_200.pkl"]
    times = []
    seqs = []
    hiddens = []
    probs_regime_init = []
    probs_obs_init = []
    probs_obs_change = []
    probs_regime_change = []

    subtitles_seq = [r"$p_{0|0}^{(0)}=0.1, p_{0|1}^{(0)}=0.9, p_{0|0}^{(1)}=0.9,  p_{0|1}^{(1)}=0.1$",
                     r"$p_{0|0}^{(0)}=0.25, p_{0|1}^{(0)}=0.75, p_{0|0}^{(1)}=0.75,  p_{0|1}^{(1)}=0.25$",
                     r"$p_{0|0}^{(0)}=0.4, p_{0|1}^{(0)}=0.6, p_{0|0}^{(1)}=0.6,  p_{0|1}^{(1)}=0.4$"
                     ]
    plot_title = r"1st Order Hierarchical HMM Samples - $p^{reg-init}=0.5, p^{reg-ch} = 0.01, p^{obs-init} = 0.5, p^{catch}=0.05$"

    for file in files_1st_order:
        sample = load_obj(results_dir + file)

        times.append(sample["sample_output"][:, 0])
        seqs.append(sample["sample_output"][:, 2])
        hiddens.append(sample["sample_output"][:, 1])

        probs_regime_init.append(sample["prob_regime_init"])
        probs_obs_init.append(sample["prob_obs_init"])
        probs_obs_change.append(sample["prob_obs_change"])
        probs_regime_change.append(sample["prob_regime_change"])


    plot_sequences(times, seqs, hiddens, subtitles_seq, plot_title,
                   save_pic=True, file_title="1st_order_seqs")

    files_2nd_order = ["2nd_5_01_5_10_200.pkl",
                       "2nd_5_01_5_25_200.pkl",
                       "2nd_5_01_5_40_200.pkl"]
    times = []
    seqs = []
    hiddens = []
    probs_regime_init = []
    probs_obs_init = []
    probs_obs_change = []
    probs_regime_change = []

    subtitles_seq = [r"$p_{0|00}^{(0)}=0.45, p_{0|01}^{(0)}=0.45, p_{0|10}^{(0)}=0.05,  p_{0|11}^{(0)}=0.05, p_{0|00}^{(1)}=0.05, p_{0|01}^{(1)}=0.05, p_{0|10}^{(1)}=0.45,  p_{0|11}^{(1)}=0.45$",
                     r"$p_{0|00}^{(0)}=0.3725, p_{0|01}^{(0)}=0.3725, p_{0|10}^{(0)}=0.125,  p_{0|11}^{(0)}=0.125, p_{0|00}^{(1)}=0.125, p_{0|01}^{(1)}=0.125, p_{0|10}^{(1)}=0.3725,  p_{0|11}^{(1)}=0.3725$",
                     r"$p_{0|00}^{(0)}=0.3, p_{0|01}^{(0)}=0.3, p_{0|10}^{(0)}=0.2,  p_{0|11}^{(0)}=0.2, p_{0|00}^{(1)}=0.2, p_{0|01}^{(1)}=0.2, p_{0|10}^{(1)}=0.3,  p_{0|11}^{(1)}=0.3$"
                     ]

    for file in files_2nd_order:
        sample = load_obj(results_dir + file)

        times.append(sample["sample_output"][:, 0])
        seqs.append(sample["sample_output"][:, 2])
        hiddens.append(sample["sample_output"][:, 1])

        probs_regime_init.append(sample["prob_regime_init"])
        probs_obs_init.append(sample["prob_obs_init"])
        probs_obs_change.append(sample["prob_obs_change"])
        probs_regime_change.append(sample["prob_regime_change"])

    plot_title = r"2nd Order Hierarchical HMM Samples - $p^{reg-init}=0.5, p^{reg-ch} = 0.01, p^{obs-init} = 0.5, p^{catch}=0.05$"
    plot_sequences(times, seqs, hiddens, subtitles_seq, plot_title,
                   save_pic=True, file_title="2nd_order_seqs_1")

    files_2nd_order_2 = ["2nd_5_01_5_200_2_1.pkl",
                       "2nd_5_01_5_200_2_2.pkl",
                       "2nd_5_01_5_200_2_3.pkl"]
    times = []
    seqs = []
    hiddens = []
    probs_regime_init = []
    probs_obs_init = []
    probs_obs_change = []
    probs_regime_change = []

    subtitles_seq = [r"$p_{0|00}^{(0)}=0.4, p_{0|01}^{(0)}=0.35, p_{0|10}^{(0)}=0.15,  p_{0|11}^{(0)}=0.1, p_{0|00}^{(1)}=0.1, p_{0|01}^{(1)}=0.15, p_{0|10}^{(1)}=0.35,  p_{0|11}^{(1)}=0.4$",
                     r"$p_{0|00}^{(0)}=0.45, p_{0|01}^{(0)}=0.4, p_{0|10}^{(0)}=0.1,  p_{0|11}^{(0)}=0.05, p_{0|00}^{(1)}=0.05, p_{0|01}^{(1)}=0.1, p_{0|10}^{(1)}=0.4,  p_{0|11}^{(1)}=0.45$",
                     r"$p_{0|00}^{(0)}=0.6, p_{0|01}^{(0)}=0.3, p_{0|10}^{(0)}=0.075,  p_{0|11}^{(0)}=0.025, p_{0|00}^{(1)}=0.025, p_{0|01}^{(1)}=0.075, p_{0|10}^{(1)}=0.3,  p_{0|11}^{(1)}=0.6$"
                     ]
    for file in files_2nd_order:
        sample = load_obj(results_dir + file)

        times.append(sample["sample_output"][:, 0])
        seqs.append(sample["sample_output"][:, 2])
        hiddens.append(sample["sample_output"][:, 1])

        probs_regime_init.append(sample["prob_regime_init"])
        probs_obs_init.append(sample["prob_obs_init"])
        probs_obs_change.append(sample["prob_obs_change"])
        probs_regime_change.append(sample["prob_regime_change"])

    plot_title = r"2nd Order Hierarchical HMM Samples - $p^{reg-init}=0.5, p^{reg-ch} = 0.01, p^{obs-init} = 0.5, p^{catch}=0.05$"
    plot_sequences(times, seqs, hiddens, subtitles_seq, plot_title,
                   save_pic=True, file_title="2nd_order_seqs_2")
    # sbl_surprise_SP = np.loadtxt(results_dir + "sbl_surprise_SP_200.txt")
    # sbl_surprise_AP = np.loadtxt(results_dir + "sbl_surprise_AP_200.txt")
    # sbl_surprise_TP = np.loadtxt(results_dir + "sbl_surprise_TP_200.txt")
    #
    # sbl_surprise_SP[np.isinf(sbl_surprise_SP)] = 0
    #
    # # Unpack array for easier reading
    # time = sbl_surprise_SP[:, 0]
    # sequence = sbl_surprise_SP[:, 1]
    # hidden = sbl_surprise_SP[:, 2]
    #
    # surprisal_SP = standardize(sbl_surprise_SP[:, 3:6])
    # surprisal_AP = standardize(sbl_surprise_AP[:, 3:6])
    # surprisal_TP = standardize(sbl_surprise_TP[:, 3:6])
    #
    # if sbl_surprise_SP.shape[1] < 9:
    #     alpha = sbl_surprise_SP[:, 6]
    #     beta = sbl_surprise_SP[:, 7]
    # else:
    #     alpha_0 = sbl_surprise_SP[:, 6]
    #     alpha_1 = sbl_surprise_SP[:, 7]
    #     beta_0 = sbl_surprise_SP[:, 8]
    #     beta_1 = sbl_surprise_SP[:, 9]
    #
    # # Input Array: [t, obs_t, hidden_t, PS_t, BS_t, CS_t] + [alphas, betas]
    # # plot_sequence(time, sequence, hidden, save_pic=False)
    # # plot_surprise(time, [predictive_surprisal, bayesian_surprisal, corrected_surprisal], "Surprisal Comparison", save_pic=False)
    #
    # plot_final(time, sequence, hidden, [surprisal_SP, surprisal_AP, surprisal_TP],
    #            alpha, beta, type="Beta-Bernoulli",
    #            save_pic=False)
    #
    # draw_animation(time, sequence, hidden,
    #                surprisal_TP,
    #                alpha, beta, save_pic=True)
