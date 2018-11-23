import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import argparse

import pickle
from scipy.io import savemat

results_dir = os.getcwd() + "/results/"
fig_dir = os.getcwd() + "/pics/"

mpl.rcParams['keymap.save'] = ''

def save_obj(obj, title):
    with open(title + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class seq_gen():
    """
    DESCRIPTION:
        * Generative model to sample sequence of binary observations (0/1)
        * Transition matrix on lowest level determines alternation of obs states - 2nd order Markovity in emissions!
    INPUT:
        * prob_regime_init: Initial probability vector for hidden state
        * prob_regime_change: Probability with which hidden state changes
        * prob_obs_init: Initial probability vector for observed state
        * prob_obs_change: Off-diag prob slow regime/On-diag prob fast regime
        - p^{i}_{o_t=0|o_t-1, o_t-2} - 4 for i=0 and 4 for i=1
        - p^{i}_{o_t=0|o_t-1=0, o_t-2=0}, p^{i}_{o_t=0|o_t-1=0, o_t-2=1},
          p^{i}_{o_t=0|o_t-1=1, o_t-2=0}, p^{i}_{o_t=0|o_t-1=1, o_t-2=1}
    OUTPUT:
        * sample: A sequence of observed states with length t
    """
    def __init__(self, order, prob_catch,
                 prob_regime_init, prob_regime_change,
                 prob_obs_init, prob_obs_change):
        self.order = order

        self.obs_space = 2
        self.regime_space = 2

        self.prob_regime_init = prob_regime_init
        self.prob_obs_init = prob_obs_init

        self.prob_obs_change = prob_obs_change
        self.prob_regime_change = prob_regime_change
        self.prob_catch = prob_catch

        self.check_input_dim()
        self.transition_matrices = self.construct_transitions()

    def check_input_dim(self):
        if len(self.prob_regime_init) != self.regime_space:
            raise ValueError("Initial regime prob vector has wrong dim")
        elif len(self.prob_obs_init) - 1 != self.obs_space:
            raise ValueError("Initial obs prob vector has wrong dim")
        elif type(self.prob_regime_change) != float or self.prob_regime_change > 1 or self.prob_regime_change < 0:
            raise ValueError("Regime change prob has to be a float between 0 and 1")
        elif type(self.prob_catch) != float or self.prob_catch > 1 or self.prob_catch < 0:
            raise ValueError("Catch probability has to be a float between 0 and 1")
        elif len(self.prob_obs_change) != 2*self.obs_space**self.order:
            raise ValueError("Need to specify {} probs for emissions".format(2*self.obs_space**self.order))
        else:
            print("All input arrays conform with the specified dimensions.")

    def construct_transitions(self):

        if self.order == 1:
            B_0 = np.array([
                [self.prob_obs_change[0] - self.prob_regime_change/4 - self.prob_catch/4,
                 1 - self.prob_obs_change[0] - self.prob_regime_change/4 - self.prob_catch/4,
                 self.prob_catch/2,
                 self.prob_regime_change/2],
                [self.prob_obs_change[1] - self.prob_regime_change/4 - self.prob_catch/4,
                 1 - self.prob_obs_change[1] - self.prob_regime_change/4 - self.prob_catch/4,
                 self.prob_catch/2,
                 self.prob_regime_change/2],
                [self.prob_obs_change[2] - self.prob_regime_change/4 - self.prob_catch/4,
                 1 - self.prob_obs_change[2] - self.prob_regime_change/4 - self.prob_catch/4,
                 self.prob_catch/2,
                 self.prob_regime_change/2],
                [self.prob_obs_change[3] - self.prob_regime_change/4 - self.prob_catch/4,
                 1 - self.prob_obs_change[3] - self.prob_regime_change/4 - self.prob_catch/4,
                 self.prob_catch/2,
                 self.prob_regime_change/2]])
            B_1 = np.array([
                [self.prob_obs_change[4] - self.prob_regime_change/4 - self.prob_catch/4,
                 1 - self.prob_obs_change[4] - self.prob_regime_change/4 - self.prob_catch/4,
                 self.prob_catch/2,
                 self.prob_regime_change/2],
                [self.prob_obs_change[5] - self.prob_regime_change/4 - self.prob_catch/4,
                 1 - self.prob_obs_change[5] - self.prob_regime_change/4 - self.prob_catch/4,
                 self.prob_catch/2,
                 self.prob_regime_change/2],
                [self.prob_obs_change[6] - self.prob_regime_change/4 - self.prob_catch/4,
                 1 - self.prob_obs_change[6] - self.prob_regime_change/4 - self.prob_catch/4,
                 self.prob_catch/2,
                 self.prob_regime_change/2],
                [self.prob_obs_change[7] - self.prob_regime_change/4 - self.prob_catch/4,
                 1 - self.prob_obs_change[7] - self.prob_regime_change/4 - self.prob_catch/4,
                 self.prob_catch/2,
                 self.prob_regime_change/2]])

        elif self.order == 2:
            B_0 = np.array([
                [self.prob_obs_change[0] - self.prob_regime_change/2 - self.prob_catch/2,
                 1 - self.prob_obs_change[0] - self.prob_regime_change/2 - self.prob_catch/2,
                 self.prob_catch,
                 self.prob_regime_change],
                [self.prob_obs_change[1] - self.prob_regime_change/2 - self.prob_catch/2,
                 1 - self.prob_obs_change[1] - self.prob_regime_change/2 - self.prob_catch/2,
                 self.prob_catch,
                 self.prob_regime_change],
                [self.prob_obs_change[2] - self.prob_regime_change/2 - self.prob_catch/2,
                 1 - self.prob_obs_change[2] - self.prob_regime_change/2 - self.prob_catch/2,
                 self.prob_catch,
                 self.prob_regime_change],
                [self.prob_obs_change[3] - self.prob_regime_change/2 - self.prob_catch/2,
                 1 - self.prob_obs_change[3] - self.prob_regime_change/2 - self.prob_catch/2,
                 self.prob_catch,
                 self.prob_regime_change]])
            B_1 = np.array([
                [self.prob_obs_change[4] - self.prob_regime_change/2 - self.prob_catch/2,
                 1 - self.prob_obs_change[4] - self.prob_regime_change/2 - self.prob_catch/2,
                 self.prob_catch,
                 self.prob_regime_change],
                [self.prob_obs_change[5] - self.prob_regime_change/2 - self.prob_catch/2,
                 1 - self.prob_obs_change[5] - self.prob_regime_change/2 - self.prob_catch/2,
                 self.prob_catch,
                 self.prob_regime_change],
                [self.prob_obs_change[6] - self.prob_regime_change/2 - self.prob_catch/2,
                 1 - self.prob_obs_change[6] - self.prob_regime_change/2 - self.prob_catch/2,
                 self.prob_catch,
                 self.prob_regime_change],
                [self.prob_obs_change[7] - self.prob_regime_change/2 - self.prob_catch/2,
                 1 - self.prob_obs_change[7] - self.prob_regime_change/2 - self.prob_catch/2,
                 self.prob_catch,
                 self.prob_regime_change]])

        print("HHMM correctly initialized. Ready to Sample.")
        transition_matrices = [B_0, B_1]
        return transition_matrices

    def get_sample_idx(self, Q, t_1, t_2):
        if Q[t_1, 1]==0 and Q[t_2, 1]==0:
            idx = 0
        elif Q[t_1, 1]==0 and Q[t_2, 1]==1:
            idx = 1
        elif Q[t_1, 1]==1 and Q[t_2, 1]==0:
            idx = 2
        elif Q[t_1, 1]==1 and Q[t_2, 1]==1:
            idx = 3
        else:
            if Q[t_1, 1]==2:
                counter = t_1-1
                while Q[counter, 1]==2 or Q[counter-1, 1]==2:
                    counter -= 1
                idx = self.get_sample_idx(Q, counter, counter-1)
            elif Q[t_2, 1]==2:
                counter = t_2-1
                while Q[counter, 1]==2:
                    counter -= 1
                idx = self.get_sample_idx(Q, t_1, counter)
        return idx

    def sample(self, seq_length):
        """
        INPUT:
            * seq_length: Length of desired observed sequence
        OUTPUT:
            * sample: (t x 4) array: index, hidden, observed, alternation indicator
        DESCRIPTION:
            1. Sample inital regime and first trial from initial vectors
            2. Loop through desired time steps
        """
        Q = np.zeros((seq_length, 2)).astype(int)
        Q[0:2, 0] = np.random.multinomial(2, self.prob_regime_init).argmax()
        Q[0:2, 1] = np.random.multinomial(2, self.prob_obs_init).argmax()

        act_regime = Q[1, 0]
        for t in range(2, seq_length):
            # Sample observed state/trial/stimulus transition
            if Q[t-1, 1] != 2:
                # Check for catch trial case
                idx = self.get_sample_idx(Q, t-1, t-2)
                Q[t, 1] = np.random.multinomial(1, self.transition_matrices[act_regime][idx, :]).argmax()

            # If regime switch is sampled on lower level - resample regime and try again
            while Q[t, 1] == 3:
                if act_regime == 0:
                    act_regime = 1
                elif act_regime == 1:
                    act_regime = 0
                idx = self.get_sample_idx(Q, t-1, t-2)
                Q[t, 1] = np.random.multinomial(1, self.transition_matrices[act_regime][idx, :]).argmax()

            # Set active regime to the one which we finally sample
            Q[t, 0] = act_regime

        Q[Q[:, 1] == 2, 0] = 2
        Q = Q.astype(float)
        Q[Q[:, 1] == 2, 1] = 0.5
        # Add column with trial/obs/time
        self.sample = np.column_stack((np.arange(seq_length), Q))
        print("Done sampling {} timesteps.".format(seq_length))
        return self.sample

def sample_and_save(seq_gen_temp, seq_length, title, matlab_out, plot_seq):
    sequence = seq_gen_temp.sample(seq_length)

    fig, ax = plt.subplots()
    coloring = []
    for j in range(len(sequence[:, 0])):
        if sequence[j, 1] == 0: coloring += "g"
        elif sequence[j, 1] == 1: coloring += "b"
        else: coloring += "r"

    ax.scatter(np.arange(seq_length), sequence[:, 2], s=4.5, c=coloring)

    def save(event, seq_length=seq_length, title=title,
             sequence=sequence, seq_gen_temp=seq_gen_temp,
             matlab_out=matlab_out):

        if event.key == 's':
            event.canvas.figure.savefig(fig_dir + title +'.png', dpi=300)
            plt.close()

            sequence[sequence[:, 2] == 0.5, 2] = 2

            sequence_meta = {"sample_output": sequence,
                             "prob_regime_init": seq_gen_temp.prob_regime_init,
                             "prob_obs_init": seq_gen_temp.prob_obs_init,
                             "prob_obs_change": seq_gen_temp.prob_obs_change,
                             "prob_regime_change": seq_gen_temp.prob_regime_change}

            if matlab_out:
                savemat(results_dir + title, sequence_meta)
            else:
                save_obj(sequence_meta, results_dir + title)
            print('Saved figure and outfiled file')

        elif event.key == "n":
            plt.close()
            sequence = seq_gen_temp.sample(seq_length)
            fig, ax = plt.subplots()
            coloring = []
            for j in range(len(sequence[:, 0])):
                if sequence[j, 1] == 0: coloring += "g"
                elif sequence[j, 1] == 1: coloring += "b"
                else: coloring += "r"

            ax.scatter(np.arange(seq_length), sequence[:, 2], s=4.5, c=coloring)
            fig.canvas.mpl_connect('key_press_event', save)
            plt.show()

    fig.canvas.mpl_connect('key_press_event', save)

    plt.show()


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
    parser.add_argument('-obs_change','--prob_obs_change', nargs='+',
                        help="Probability of changing regime", action="store", type=float)
    parser.add_argument('-catch', '--prob_catch', action="store",
                        default=0.05, type=float,
                        help="Probability of changing regime")
    parser.add_argument('-t', '--title', action="store",
                        default="temporary_sample_title", type=str,
						help='Title of file which stores sequence')
    parser.add_argument('-seq', '--sequence_length', action="store",
                        default=200, type=int,
						help='Length of binary sequence being processed')
    parser.add_argument('-matlab', '--mat_file_out',
                        action="store_true",
                        default=True,
						help='Save output as a .mat file')
    parser.add_argument('-order', '--markov_order', action="store",
                        default=1, type=int,
						help='Markov dependency on observation level')
    parser.add_argument('-p', '--plot_seq',
                        action="store_true",
                        default=False,
						help='View/Plot the sampled sequence')


    args = parser.parse_args()

    prob_regime_init = np.array([args.prob_regime_init, 1-args.prob_regime_init])
    prob_regime_change = args.prob_regime_change
    prob_obs_init = np.array([args.prob_obs_init, 1-args.prob_obs_init, 0])
    prob_obs_change = args.prob_obs_change
    prob_catch = args.prob_catch

    order = args.markov_order
    seq_length = args.sequence_length
    title = args.title
    matlab_out = args.mat_file_out
    plot_seq = args.plot_seq

    gen_temp = seq_gen(order, prob_catch, prob_regime_init, prob_regime_change,
                       prob_obs_init, prob_obs_change)
    sample_and_save(gen_temp, seq_length, title, matlab_out, plot_seq)

    """
    pythonw seq_gen.py -t 2nd_5_01_5_10_200 -obs_change 0.45 0.45 0.05 0.05 0.05 0.05 0.45 0.45 -order 2 - matlab
    """
