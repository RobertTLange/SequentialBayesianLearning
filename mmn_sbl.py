import numpy as np
from scipy.special import gamma, digamma

from mmn_seq_gen import hhmm


class BB_SBL():
    """
    DESCRIPTION:
    INPUT:
    OUTPUT:
    """
    def __init__(self, seq, tau):
        # Initialize SBL-learned sequence and exponential forgetting parameter
        self.sequence = seq
        self.t = 0
        self.T = len(seq)
        self.tau = tau

        # Generate T-dim vector indicating no-alternation from t-1 to t
        self.repetition = np.zeros(self.T)
        for t in range(1, self.T):
            if self.sequence[t] == self.sequence[t-1]:
                self.repetition[t] = 1

        # Generate one T matrix with all discounting values
        self.exp_forgetting = np.exp(-self.tau*np.arange(self.T)[::-1])

        # Initialize parameters for beta prior
        self.alpha = 1
        self.beta = 1

    def update_posterior(self, t, type):
        exp_weighting = self.exp_forgetting[-t:]
        if type == "SP":
            self.alpha = 1 + np.dot(exp_weighting, self.sequence[:t])
            self.beta = 1 + np.dot(exp_weighting, 1-self.sequence[:t])
        elif type == "AP":
            self.alpha = 1 + np.dot(exp_weighting, self.alternation[:t])
            self.beta = 1 + np.dot(exp_weighting, 1-self.alternation[:t])
        elif type == "TP":
            self.alpha = 1
            self.beta = 1

    def compute_surprisal(self, type):
        print("Computing different surprisal measures.")
        PS = np.zeros(self.T - 1)
        BS = np.zeros(self.T - 1)
        CS = np.zeros(self.T - 1)

        for t in range(1, self.T):
            # Loop over the full sequence and compute surprisal iteratively
            PS[t] = predictive_surprisal(type, t)
            BS[t] = bayesian_surprisal(type, t)
            CS[t] = corrected_surprisal(type, t)

        print("Done computing predictive, bayesian and confidence-corrected surprisal.")
        return PS, BS, CS

    def predictive_surprisal(self, t):
        if type == "SP":
            update_posterior(self, t-1, "SP")
            PS = -np.log(((self.alpha/(self.alpha + self.beta))**self.seq[t])*((1 - self.alpha/(self.alpha + self.beta))**(1-self.seq[t])))
        elif type == "AP":
            PS = 1
        elif type == "TP":
            PS = 1
        return PS

    def bayesian_surprisal(self):
        if self.type == "SP":
            BS = 1
        elif self.type == "AP":
            BS = 1
        elif self.type == "TP":
            BS = 1
        return BS

    def corrected_surprisal(self):
        if self.type == "SP":
            CS = 1
        elif self.type == "AP":
            CS = 1
        elif self.type == "TP":
            CS = 1
        return CS


class GRW_SBL():

    def __init__(self, seq, tau, type):
        self.seq = seq
        self.tau = tau
        self.type = type

    def compute_surprisal(self):
        PS = predictive_surprisal()
        BS = bayesian_surprisal()
        CS = corrected_surprisal()
        return

    def predictive_surprisal(self):
        PS = 1
        return PS

    def bayesian_surprisal(self):
        if self.type == "SP":
            BS = 1
        elif self.type == "AP":
            BS = 1
        return BS

    def corrected_surprisal(self):
        if self.type == "SP":
            CS = 1
        elif self.type == "AP":
            CS = 1
        return CS


if __name__ == "__main__":
    hhmm_temp = hhmm(prob_regime_init=np.array([0.5, 0.5]),
                     prob_regime_change=0.01,
                     prob_obs_init=np.array([0.5, 0.5, 0]),
                     prob_obs_change=0.25)
    hhmm_seq = hhmm_temp.sample_seq(10)[:, 2]

    BB_SBL_temp = BB_SBL(hhmm_seq, tau=0.)
    BB_SBL_temp.update_posterior(3, "SP")
    print(BB_SBL_temp.alpha, BB_SBL_temp.beta)
