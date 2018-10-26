import argparse
import numpy as np
from scipy.special import gamma, digamma
from hhmm_seq_gen import hhmm


class SBL_BB():
    """
    DESCRIPTION:
    INPUT:
    OUTPUT:
    """
    def __init__(self, seq, tau):
        # Initialize SBL-learned sequence and exponential forgetting parameter
        self.sequence = seq[:, 1]
        self.hidden = seq[:, 0]
        self.T = len(seq)
        self.tau = tau

        self.no_obs = 2

        # AP: Generate T-dim vector indicating no-alternation from t-1 to t
        self.repetition = np.zeros(self.T)
        for t in range(1, self.T):
            if self.sequence[t] == self.sequence[t-1]:
                self.repetition[t] = 1

        # TP: Generate T-dim vectors indicating transition from state i
        self.transitions = np.zeros((self.T, self.no_obs))
        for t in range(1, self.T):
            self.transitions[t, 0] = (self.sequence[t-1] == 0)
            self.transitions[t, 1] = 1 - self.transitions[t, 0]

        # Generate one T matrix with all discounting values
        self.exp_forgetting = np.exp(-self.tau*np.arange(self.T)[::-1])

    def update_posterior(self, t, type):
        exp_weighting = self.exp_forgetting[-t:]

        if type == "SP":
            self.alpha = 1 + np.dot(exp_weighting, self.sequence[:t])
            self.beta = 1 + np.dot(exp_weighting, 1-self.sequence[:t])
        elif type == "AP":
            self.alpha = 1 + np.dot(exp_weighting, self.repetition[1:(t+1)])
            self.beta = 1 + np.dot(exp_weighting, 1-self.repetition[1:(t+1)])
        elif type == "TP":
            # print(self.sequence[:t], self.transition_from_0[:t], self.transition_from_1[:t])
            self.alpha = np.empty(self.no_obs)
            self.beta = np.empty(self.no_obs)

            for i in range(self.no_obs):
                self.alpha[i] = 1 + np.dot(exp_weighting, self.sequence[:t]*self.transitions[1:(t+1), i])
                self.beta[i] = 1 + np.dot(exp_weighting, (1 - self.sequence[:t])*self.transitions[1:(t+1), i])

    def compute_surprisal(self, type):
        print("{}: Computing different surprisal measures for all {} timesteps.".format(type, self.T))

        results = []

        for t in range(2, self.T):
            # Loop over the full sequence and compute surprisal iteratively
            self.update_posterior(t-1, type)
            PS_temp = self.predictive_surprisal(t, type)
            BS_temp = self.bayesian_surprisal(t, type)
            CS_temp = self.corrected_surprisal(t, type)
            temp = [t, self.sequence[t], self.hidden[t], PS_temp, BS_temp, CS_temp]
            distr_params = [item for sublist in [self.alpha.tolist(), self.beta.tolist()] for item in sublist]
            results.append(temp + distr_params)
        print("{}: Done computing surprisal measures for all {} timesteps.".format(type, self.T))
        return np.asarray(results)

    def predictive_surprisal(self, t, type):
        if type == "SP":
            PS = -np.log(((self.alpha/(self.alpha + self.beta))**self.sequence[t]) \
                         *((1 - self.alpha/(self.alpha + self.beta))**(1-self.sequence[t])))
        elif type == "AP":
            PS = -np.log(((self.alpha/(self.alpha + self.beta))**self.repetition[t]) \
                         *((1 - self.alpha/(self.alpha + self.beta))**(1-self.repetition[t])))
        elif type == "TP":
            PS = 0
            for i in range(self.transitions.shape[1]):
                PS += -self.transitions[t, i]*\
                       np.log((self.alpha[i]/(self.alpha[i] + self.beta[i]))**self.sequence[t]*\
                              (1 - self.alpha[i]/(self.alpha[i] + self.beta[i]))**(1 - self.sequence[t]))
        return PS

    def bayesian_surprisal(self, t, type):
        if type == "SP":
            BS = np.log(gamma(self.alpha + self.beta)/(gamma(self.alpha) * gamma(self.beta))) \
                 - np.log(gamma(self.alpha + self.beta + 1)/(gamma(self.alpha + self.sequence[t]) * gamma(self.beta + 1 - self.sequence[t]))) \
                 - self.sequence[t]*digamma(self.alpha) + (self.sequence[t] - 1)*digamma(self.beta) + digamma(self.alpha + self.beta)
        elif type == "AP":
            BS = np.log(gamma(self.alpha + self.beta)/(gamma(self.alpha) * gamma(self.beta))) \
                 - np.log(gamma(self.alpha + self.beta + 1)/(gamma(self.alpha + self.repetition[t]) * gamma(self.beta + 1 - self.repetition[t]))) \
                 - self.repetition[t]*digamma(self.alpha) + (self.repetition[t] - 1)*digamma(self.beta) + digamma(self.alpha + self.beta)
        elif type == "TP":
            BS = 0
            for i in range(self.transitions.shape[1]):
                BS += self.transitions[t, i]*\
                      (np.log(gamma(self.alpha[i] + self.beta[i])/(gamma(self.alpha[i]) * gamma(self.beta[i])))\
                       - np.log(gamma(self.alpha[i] + self.beta[i] + 1)/(gamma(self.alpha[i] + self.sequence[t]) * gamma(self.beta[i] + 1 - self.sequence[t]))) \
                       - self.sequence[t]*digamma(self.alpha[i]) + (self.sequence[t] - 1)*digamma(self.beta[i]) + digamma(self.alpha[i] + self.beta[i]))
        return BS

    def corrected_surprisal(self, t, type):
        if type == "SP":
            CS = np.log(gamma(self.alpha + self.beta)/(gamma(self.alpha) * gamma(self.beta))) - np.log(2) \
                 + (self.alpha - 1 - self.sequence[t])*digamma(self.alpha) \
                 + (self.beta - 2 + self.sequence[t])*digamma(self.beta) \
                 + (3 - self.alpha - self.beta)*digamma(self.alpha + self.beta)
        elif type == "AP":
            CS = np.log(gamma(self.alpha + self.beta)/(gamma(self.alpha) * gamma(self.beta))) - np.log(2) \
                 + (self.alpha - 1 - self.repetition[t])*digamma(self.alpha) \
                 + (self.beta - 2 + self.repetition[t])*digamma(self.beta) \
                 + (3 - self.alpha - self.beta)*digamma(self.alpha + self.beta)
        elif type == "TP":
            CS = 0
            for i in range(self.transitions.shape[1]):
                CS += self.transitions[t, i]*\
                      (np.log(gamma(self.alpha[i] + self.beta[i])/(gamma(self.alpha[i]) * gamma(self.beta[i])))\
                       - np.log(2) + (self.alpha[i] - 1 - self.sequence[t])*digamma(self.alpha[i])\
                       + (self.beta[i] - 2 - self.sequence[t])*digamma(self.beta[i]) +(3 - self.alpha[i] - self.beta[i])*digamma(self.alpha[i] + self.beta[i]))
        return CS


def main(prob_regime_init, prob_regime_change,
         prob_obs_init, prob_obs_change, seq_length,
         tau, model, save_results):
    # I: Generate binary sequence sampled from HHMM
    hhmm_temp = hhmm(prob_regime_init, prob_regime_change,
                     prob_obs_init, prob_obs_change)
    hhmm_seq = hhmm_temp.sample_seq(seq_length)[:, [1,2]]

    # II: Compute Surprisal for all time steps for Stimulus Prob BB Model
    BB_SBL_temp = SBL_BB(hhmm_seq, tau)
    results = BB_SBL_temp.compute_surprisal(model)

    if save_results:
        title = "sbl_surprise_" + str(model) + "_" + str(seq_length) + "_" + str(tau)
        np.savetxt(title, results)


def test_agent(prob_regime_init, prob_regime_change,
               prob_obs_init, prob_obs_change, seq_length,
               tau, model):
    # Test I: Generate binary sequence sampled from HHMM
    hhmm_temp = hhmm(prob_regime_init, prob_regime_change,
                     prob_obs_init, prob_obs_change)
    hhmm_seq = hhmm_temp.sample_seq(seq_length)[:, [1,2]]

    # Test IIa: Initialize SBL (seq, forgetting param), update posterior (t=3)
    BB_SBL_temp = SBL_BB(hhmm_seq, tau=0.)
    BB_SBL_temp.update_posterior(1, model)
    print("{}: Beta-Distribution after 1 timestep: alpha = {}, beta = {}".format(model, BB_SBL_temp.alpha, BB_SBL_temp.beta))
    print("---------------------------------------------")

    # Test IIb: Compute Surprisal once (SP, t=3)
    print("{}: Predictive Surprisal at t=2: {}".format(model, BB_SBL_temp.predictive_surprisal(2, model)))
    print("{}: Bayesian Surprisal at t=2: {}".format(model, BB_SBL_temp.bayesian_surprisal(2, model)))
    print("{}: Confidence-Corrected Surprisal at t=2: {}".format(model, BB_SBL_temp.corrected_surprisal(2, model)))
    print("---------------------------------------------")

    # Test IIc: Compute Surprisal for all time steps for Stimulus Prob BB Model
    results = BB_SBL_temp.compute_surprisal(model)
    print("---------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg_init', '--prob_regime_init', action="store", default=0.5, type=float,
						help="Initial regime probability")
    parser.add_argument('-reg_change', '--prob_regime_change', action="store", default=0.01, type=float,
						help="Probability of changing regime")
    parser.add_argument('-obs_init', '--prob_obs_init', action="store", default=0.5, type=float,
						help="Initial regime probability")
    parser.add_argument('-obs_change', '--prob_obs_change', action="store", default=0.25, type=float,
						help="Probability of changing regime")
    parser.add_argument('-seq', '--sequence_length', action="store", default=200, type=int,
						help='Length of binary sequence being processed')
    parser.add_argument('-tau', '--forget_param', action="store", default=0., type=float,
                        help='Exponentially weighting parameter for memory/posterior updating')
    parser.add_argument('-model', '--model', action="store", default="SP", type=str,
                        help='Beta-Bernoulli Probability Model (SP, AP, TP)')
    parser.add_argument('-T', '--test', action="store_true", help='Run tests.')
    parser.add_argument('-S', '--save', action="store_true", help='Save results to array.')

    args = parser.parse_args()

    prob_regime_init = np.array([args.prob_regime_init, 1-args.prob_regime_init])
    prob_regime_change = args.prob_regime_change
    prob_obs_init = np.array([args.prob_obs_init, 1-args.prob_obs_init, 0])
    prob_obs_change = args.prob_obs_change

    seq_length = args.sequence_length
    tau = args.forget_param
    model = args.model

    run_test = args.test
    save_results = args.save

    if run_test:
        print("Started running basic tests.")
        test_agent(prob_regime_init, prob_regime_change,
                   prob_obs_init, prob_obs_change, seq_length,
                   tau, model)

    else:
        main(prob_regime_init, prob_regime_change,
             prob_obs_init, prob_obs_change, seq_length,
             tau, model, save_results)

    """
    python mmn_sbl.py -model SP
    """