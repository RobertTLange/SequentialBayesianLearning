import os
import pickle
from scipy import log, log2, array, zeros
import scipy.io as sio
from scipy.special import gamma, digamma, gammaln
from scipy.stats import dirichlet

import numpy as np

results_dir = os.getcwd() + "/results/"
fig_dir = os.getcwd() + "/figures/"


def save_obj(obj, title):
    with open(title + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(title):
    filename, file_extension = os.path.splitext(title)
    if file_extension == ".mat":
        return sio.loadmat(title)
    else:
        with open(title, 'rb') as f:
            return pickle.load(f)


# def kl_general(p, q):
#     """Compute the KL divergence between two discrete probability distributions
#     The calculation is done directly using the Kullback-Leibler divergence,
#     KL( p || q ) = sum_{x} p(x) ln( p(x) / q(x) )
#     Natural logarithm is used!
#     """
#     if (p==0.).sum()+(q==0.).sum() > 0:
#         raise Exception, "Zero bins found"
#     return (p*(np.log(p) - np.log(q))).sum()


def kl_dir(alphas, betas):
    """Compute the KL divergence between two Dirichlet probability distributions
    """
    alpha_0 = alphas.sum()
    beta_0 = betas.sum()

    a_part = gammaln(alpha_0) - (gammaln(alphas)).sum()
    b_part = gammaln(beta_0) - (gammaln(betas)).sum()

    ab_part = ((alphas - betas)*(digamma(alphas) - digamma(alpha_0))).sum()
    return a_part - b_part + ab_part


def draw_dirichlet_params(alphas):
    if len(alphas) != 8:
        raise ValueError("Provide correct size of concentration params")
    return np.random.dirichlet((alphas), 1).transpose()


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

def stand(surprise):
    arr = np.array(surprise)
    temp = arr/np.nanmax(arr,axis=0)
    return temp

def preproc_surprisal(SP, AP, TP):
    time = SP["time"]
    hidden = SP["hidden"]
    sequence = SP["sequence"]

    PS = [stand(SP["predictive_surprise"]),
          stand(AP["predictive_surprise"]),
          stand(TP["predictive_surprise"])]
    BS = [stand(SP["bayesian_surprise"]),
          stand(AP["bayesian_surprise"]),
          stand(TP["bayesian_surprise"])]
    CS = [stand(SP["confidence_corrected_surprise"]),
          stand(AP["confidence_corrected_surprise"]),
          stand(TP["confidence_corrected_surprise"])]

    return time, hidden, sequence, PS, BS, CS


if __name__ == "__main__":
    alphas_old = np.array([2, 1,  1])
    alphas = np.array([3, 1,  1])
    # Why is Kl negative in bits different? - divergence non-neg by Gibbs!!!
    print(kl_dir(alphas_old, alphas))
