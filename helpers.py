import os
import pickle
from scipy import log, log2, array, zeros
import scipy.io as sio
from scipy.special import gamma, digamma, gammaln
from scipy.stats import dirichlet

import numpy as np

results_dir = os.getcwd() + "/results/"
fig_dir = os.getcwd() + "/pics/"

def save_obj(obj, title):
    with open(title + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(title, matlab_in):
    if matlab_in:
        return sio.loadmat(title)
    else:
        with open(title, 'rb') as f:
            return pickle.load(f)

def kl_general(p, q):
    """Compute the KL divergence between two discrete probability distributions
    The calculation is done directly using the Kullback-Leibler divergence,
    KL( p || q ) = sum_{x} p(x) ln( p(x) / q(x) )
    Natural logarithm is used!
    """
    if (p==0.).sum()+(q==0.).sum() > 0:
        raise Exception, "Zero bins found"
    return (p*(np.log(p) - np.log(q))).sum()

def kl_dir(alphas, betas):
    """Compute the KL divergence between two Dirichlet probability distributions
    """
    alpha_0 = alphas.sum()
    beta_0 = betas.sum()
    a_part = gammaln(alpha_0) - (gammaln(alphas)).sum()
    b_part = gammaln(beta_0) - (gammaln(betas)).sum()
    ab_part = ((alphas - betas)*digamma(alphas - alpha_0)).sum()
    return a_part - b_part + ab_part


if __name__ == "__main__":
    alphas = np.array([225, 258,  18])
    betas = np.array([226, 258,  18])
    # Why is Kl negative in bits different? - divergence non-neg by Gibbs!!!
    print(kl_dir(alphas, betas))

    def calculate_KL_beta(alpha_0, alpha_1, beta_0, beta_1):
        return np.log(gamma(alpha_0)) - np.log(gamma(alpha_1)) + np.log(gamma(beta_1)) + (alpha_1 - beta_1)*(digamma(alpha_1) - digamma(alpha_0))

    print(calculate_KL_beta(225, 258, 226, 258))
