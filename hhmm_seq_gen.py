import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import pickle

results_dir = os.getcwd() + "/results/"

def save_obj(obj, title):
    with open(title + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class hhmm_1st():
    """
    DESCRIPTION:
        * 2-Layer Hierarchical Hidden Markov Model
        * Can be used to sample sequence of binary observations (0/1)
        * Transition matrix on lowest level determines alternation of obs states
    INPUT:
        * prob_regime_init: Initial probability vector for hidden state
        * prob_regime_change: Probability with which hidden state changes
        * prob_obs_init: Initial probability vector for observed state
        * prob_obs_change: Off-diag prob slow regime/On-diag prob fast regime
    OUTPUT:
        * sample_seq: A sequence of observed states with length t
    """
    def __init__(self,
                 prob_regime_init, prob_regime_change,
                 prob_obs_init, prob_obs_change):
        self.D = 2
        self.obs_space = 2
        self.regime_space = 2

        self.prob_regime_init = prob_regime_init
        self.prob_obs_init = prob_obs_init

        self.prob_obs_change = prob_obs_change
        self.prob_regime_change = prob_regime_change

        self.check_input_dim()
        self.transition_matrices = self.construct_transitions()

    def check_input_dim(self):
        if len(self.prob_regime_init) != self.regime_space:
            raise ValueError("Arrays must have the same size")
        elif len(self.prob_obs_init) - 1 != self.obs_space:
            raise ValueError("Arrays must have the same size")
        else:
            print("All input arrays conform with the specified dimensions.")

    def construct_transitions(self):
        A_1 = np.fliplr(np.eye(self.regime_space))
        temp = 1 - self.prob_obs_change
        A_2_1 = np.array([[temp - self.prob_regime_change/2, 1 - temp - self.prob_regime_change/2, self.prob_regime_change],
                         [1 - temp - self.prob_regime_change/2, temp - self.prob_regime_change/2, self.prob_regime_change],
                         [0, 0, 0]])
        A_2_2 = np.array([[1 - temp - self.prob_regime_change/2, temp - self.prob_regime_change/2, self.prob_regime_change],
                         [temp - self.prob_regime_change/2, 1 - temp - self.prob_regime_change/2, self.prob_regime_change],
                         [0, 0, 0]])
        print("HHMM correctly initialized. Ready to Sample.")
        transition_matrices = {}
        transition_matrices[0] = A_1
        transition_matrices[1] = [A_2_1, A_2_2]
        return transition_matrices

    def sample_seq(self, seq_length):
        """
        INPUT:
            * seq_length: Length of desired observed sequence
        OUTPUT:
            * sample: (t x 4) array: index, hidden, observed, alternation indicator
        DESCRIPTION:
            1. Sample inital regime and first trial from initial vectors
            2. Loop through desired time steps
        """
        Q = np.zeros((seq_length, self.D + 1)).astype(int)
        Q[0, 0] = np.random.multinomial(1, self.prob_regime_init).argmax()
        Q[0, 1] = np.random.multinomial(1, self.prob_obs_init).argmax()

        act_regime = Q[0, 0]
        for t in range(1, seq_length):
            # Sample observed state/trial/stimulus transition
            Q[t, 1] = np.random.multinomial(1, self.transition_matrices[1][act_regime][Q[t-1, 1], :]).argmax()

            # If regime switch is sampled on lower level - resample regime and try again
            while Q[t, 1] == 2:
                act_regime = int(np.random.multinomial(1, self.transition_matrices[0][Q[t-1, 0], :]).argmax())
                Q[t, 1] = np.random.multinomial(1, self.transition_matrices[1][int(act_regime)][Q[t-1, 1], :]).argmax()

            # Set active regime to the one which we finally sample
            Q[t, 0] = act_regime

            # Check if sampled observed state equal to previous (alternation)
            if Q[t-1, 1] == Q[t-1, 1]:
                Q[t, 2] = 1

        # Add column with trial/obs/time
        self.sample = np.column_stack((np.arange(seq_length), Q))
        print("Done sampling {} timesteps.".format(seq_length))
        return self.sample


    def plot_sample(self, save=False):
        plt.scatter(self.sample[:, 0], self.sample[:, 2],
                    s=0.5, c=self.sample[:, 1])
        plt.title("HHMM-Sampled Stimulus Sequence")

        if save:
            plt.savefig("hhmm_stimulus_seq.png", dpi=300)
        else:
            plt.show()

    def sample_and_save(self, seq_length, title):
        sequence = self.sample_seq(seq_length)
        sequence_meta = {"sample_output": sequence,
                         "prob_regime_init": self.prob_regime_init,
                         "prob_obs_init": self.prob_obs_init,
                         "prob_obs_change": self.prob_obs_change,
                         "prob_regime_change": self.prob_regime_change}
        save_obj(sequence_meta, results_dir + title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg_init', '--prob_regime_init', action="store",
                        default=0.5, type=float,
                        help="Initial regime probability")
    parser.add_argument('-reg_change', '--prob_regime_change', action="store",
                        default=0.01, type=float,
                        help="Probability of changing regime")
    parser.add_argument('-obs_init', '--prob_obs_init', action="store",
                        default=0.5, type=float,
                        help="Initial regime probability")
    parser.add_argument('-obs_change', '--prob_obs_change', action="store",
                        default=0.25, type=float,
                        help="Probability of changing regime")
    parser.add_argument('-t', '--title', action="store",
                        default="temporary_sample_title", type=str,
						help='Title of file which stores sequence')
    parser.add_argument('-seq', '--sequence_length', action="store",
                        default=200, type=int,
						help='Length of binary sequence being processed')


    args = parser.parse_args()

    prob_regime_init = np.array([args.prob_regime_init, 1-args.prob_regime_init])
    prob_regime_change = args.prob_regime_change
    prob_obs_init = np.array([args.prob_obs_init, 1-args.prob_obs_init, 0])
    prob_obs_change = args.prob_obs_change

    seq_length = args.sequence_length
    title = args.title

    hhmm_temp = hhmm_1st(prob_regime_init, prob_regime_change,
                         prob_obs_init, prob_obs_change)
    hhmm_temp.sample_and_save(seq_length, title)
