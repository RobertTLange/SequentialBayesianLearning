import os
import numpy as np
import scipy.stats as ss
from scipy.special import gamma, digamma

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.style.use('ggplot')

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
            #surprisal_temp = smooth(surp[:t], degree=1)
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
    sbl_surprise_SP = np.loadtxt(results_dir + "sbl_surprise_SP_200.txt")
    sbl_surprise_AP = np.loadtxt(results_dir + "sbl_surprise_AP_200.txt")
    sbl_surprise_TP = np.loadtxt(results_dir + "sbl_surprise_TP_200.txt")

    sbl_surprise_SP[np.isinf(sbl_surprise_SP)] = 0

    # Unpack array for easier reading
    time = sbl_surprise_SP[:, 0]
    sequence = sbl_surprise_SP[:, 1]
    hidden = sbl_surprise_SP[:, 2]

    surprisal_SP = standardize(sbl_surprise_SP[:, 3:6])
    surprisal_AP = standardize(sbl_surprise_AP[:, 3:6])
    surprisal_TP = standardize(sbl_surprise_TP[:, 3:6])

    if sbl_surprise_SP.shape[1] < 9:
        alpha = sbl_surprise_SP[:, 6]
        beta = sbl_surprise_SP[:, 7]
    else:
        alpha_0 = sbl_surprise_SP[:, 6]
        alpha_1 = sbl_surprise_SP[:, 7]
        beta_0 = sbl_surprise_SP[:, 8]
        beta_1 = sbl_surprise_SP[:, 9]

    # Input Array: [t, obs_t, hidden_t, PS_t, BS_t, CS_t] + [alphas, betas]
    # plot_sequence(time, sequence, hidden, save_pic=False)
    # plot_surprise(time, [predictive_surprisal, bayesian_surprisal, corrected_surprisal], "Surprisal Comparison", save_pic=False)

    plot_final(time, sequence, hidden, [surprisal_SP, surprisal_AP, surprisal_TP],
               alpha, beta, type="Beta-Bernoulli",
               save_pic=False)

    draw_animation(time, sequence, hidden,
                   surprisal_TP,
                   alpha, beta, save_pic=True)
