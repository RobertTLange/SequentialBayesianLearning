import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import argparse

import pickle
from scipy.io import savemat

from sampling.seq_analysis import *
from utils.helpers import *

mpl.rcParams['keymap.save'] = ''


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
        - For 2nd order Markov dependency:
            - p^{i}_{o_t=0|o_t-1, o_t-2} - 4 for i=0 and 4 for i=1
            - p^{i}_{o_t=0|o_t-1=0, o_t-2=0}, p^{i}_{o_t=0|o_t-1=0, o_t-2=1},
              p^{i}_{o_t=0|o_t-1=1, o_t-2=0}, p^{i}_{o_t=0|o_t-1=1, o_t-2=1}
    OUTPUT:
        * sample: A sequence of observed states with length t
    """
    def __init__(self, order, prob_catch,
                 prob_regime_init, prob_regime_change,
                 prob_obs_init, prob_obs_change, verbose):
        # Initialize parameters of sequence generation instance
        self.order = int(order)

        self.obs_space = 2
        self.regime_space = 2

        self.prob_regime_init = prob_regime_init
        self.prob_obs_init = prob_obs_init

        self.prob_obs_change = prob_obs_change
        self.prob_regime_change = prob_regime_change
        self.prob_catch = prob_catch
        self.verbose = verbose

        # Check consistency of length of inputs
        self.check_input_dim()
        # Construct the transition matrices based on input parameters
        self.transition_matrices = self.construct_transitions()

    def check_input_dim(self):
        # Function checks if the input parameters conform with required shapes
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
            if self.verbose:
                print("All input arrays conform with the specified dimensions.")

    def construct_transitions(self):
        # Function constructs 2-regime transition matrices and checks row-stoch
        B_0 = np.zeros((self.obs_space**self.order, self.obs_space + 2))
        B_1 = np.zeros((self.obs_space**self.order, self.obs_space + 2))

        for i in range(B_0.shape[0]):
            B_0[i, 0] = self.prob_obs_change[i] - self.prob_catch/2 - self.prob_regime_change/2
            B_0[i, 1] = 1 - self.prob_obs_change[i] - self.prob_catch/2 - self.prob_regime_change/2
            B_0[i, 2] = self.prob_catch
            B_0[i, 3] = self.prob_regime_change

            B_1[i, 0] = self.prob_obs_change[i + B_0.shape[0]] - self.prob_catch/2 - self.prob_regime_change/2
            B_1[i, 1] = 1 - self.prob_obs_change[i + B_0.shape[0]] - self.prob_catch/2 - self.prob_regime_change/2
            B_1[i, 2] = self.prob_catch
            B_1[i, 3] = self.prob_regime_change

        if (np.sum(B_0, axis=1) != np.ones(B_0.shape[0])).all() or (np.sum(B_1, axis=1) != np.ones(B_1.shape[0])).all():
            raise ValueError("Matrices are not row stochastic")

        if self.verbose:
            print("HHMM correctly initialized. Ready to Sample.")
            print("--------------------------------------------")
            if self.order == 1:
                print("1st Order Transition Prob. \n Regime 0: p(0|0)={}, p(0|1)={} \n Regime 1: p(0|0)={}, p(0|1)={}".format(*self.prob_obs_change))
            if self.order == 2:
                print("2nd Order Transition Prob. \n Regime 0: p(0|00)={}, p(0|01)={}, p(0|10)={},  p(0|11)={} \n Regime 1: p(0|00)={}, p(0|01)={}, p(0|10)={},  p(0|11)={}".format(*self.prob_obs_change))
            print("--------------------------------------------")
        return [B_0, B_1]

    def get_sample_idx(self, Q, t_1, t_2):
        """
        INPUTS: Previous sequence array Q, index of previous time steps obs
            - Always pass two time steps even if order is one
        OUTPUT: Markov-order dependent row sampling index for next obs
        """
        if self.order == 1:
            if Q[t_1, 1] == 0:
                # E.g. prev obs was 0 - sample from row 0 next
                idx = 0
            elif Q[t_1, 1] == 1:
                idx = 1
            elif Q[t_1, 1] == 2:  # If prev one was catch recurse until not
                counter = t_1
                while Q[counter, 1] == 2:
                    counter -= 1
                idx = self.get_sample_idx(Q, counter, counter-1)
            else:
                raise ValueError("Wrong index - regime switches != obs!")

        if self.order == 2:
            if Q[t_1, 1] == 0 and Q[t_2, 1] == 0:
                # E.g. prev obs was 0 and before was also 0 - sample row 0 next
                idx = 0
            elif Q[t_1, 1] == 0 and Q[t_2, 1] == 1:
                idx = 1
            elif Q[t_1, 1] == 1 and Q[t_2, 1] == 0:
                idx = 2
            elif Q[t_1, 1] == 1 and Q[t_2, 1] == 1:
                idx = 3
            else:
                if Q[t_1, 1] == 2: # If prev one was catch recurse until not
                    counter = t_1
                    while Q[counter, 1] == 2 or Q[counter-1, 1] == 2:
                        counter -= 1
                    idx = self.get_sample_idx(Q, counter, counter-1)
                elif Q[t_2, 1]==2: # If prev-prev one was catch recurse until not
                    counter = t_2
                    while Q[counter, 1] == 2:
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

        # Sample first states and observations uniformly
        Q[0:self.order, 0] = np.random.multinomial(self.order, self.prob_regime_init).argmax()
        Q[0:self.order, 1] = np.random.multinomial(self.order, self.prob_obs_init).argmax()
        # Set the first active regime
        act_regime = Q[self.order, 0]

        # Run sampling over the whole sequence
        for t in range(self.order, seq_length):
            # Check if previous trial is catch
            idx = self.get_sample_idx(Q, t-1, t-2)
            Q[t, 1] = np.random.multinomial(1, self.transition_matrices[act_regime][idx, :]).argmax()

            # If regime switch is sampled - switch act_regime and try again
            while Q[t, 1] == 3:
                if act_regime == 0:
                    act_regime = 1
                elif act_regime == 1:
                    act_regime = 0
                idx = self.get_sample_idx(Q, t-1, t-2)
                Q[t, 1] = np.random.multinomial(1, self.transition_matrices[act_regime][idx, :]).argmax()

            # Set active regime to the one which we finally sample
            Q[t, 0] = act_regime

        # Switch hidden state to 2 if catch trial is sampled
        Q[Q[:, 1] == 2, 0] = 2
        # Change catch trial to 0.5 instead of 2 for nice plotting
        Q = Q.astype(float)
        Q[Q[:, 1] == 2, 1] = 0.5

        # Add column with trial/obs/time
        self.sample_seq = np.column_stack((np.arange(seq_length), Q))

        if self.verbose:
            calc_stats(self.sample_seq, self.verbose)
        return self.sample_seq


def save(sequence, seq_gen_temp, matlab_out):

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
    print('Saved data and outfiled file')


def sample_and_save(seq_gen_temp, seq_length, title, matlab_out, plot_seq):
    sequence = seq_gen_temp.sample(seq_length)
    stats, reg_0s, reg_1s = calc_stats(sequence, False)

    if plot_seq:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), dpi=300)
        fig.tight_layout()
        coloring = []
        for j in range(len(sequence[:, 0])):
            if sequence[j, 1] == 0: coloring += "g"
            elif sequence[j, 1] == 1: coloring += "b"
            else: coloring += "r"

        ax[0].scatter(np.arange(200), sequence[:200, 2], s=4.5, c=coloring)
        ax[0].set_title("First 200 Trials of Length {} Block".format(sequence.shape[0]))

        # Add extra info as additional lines with label in legend
        ax[1].hist(reg_0s[:, 2], density=True, label=r"Regime 0 ($s_t=0$)", alpha=0.5, range=(0, 20), color="g")
        ax[1].hist(reg_1s[:, 2], density=True, label=r"Regime 1 ($s_t=1$)", alpha=0.5, range=(0, 20), color="b")
        # Add extra info as additional lines with label in legend
        ax[1].plot([], [], ' ', label=r"$p(o_t = 1|s_t = 0)$: {}".format(round(stats["emp_reg0_sp"], 3)))
        ax[1].plot([], [], ' ', label=r"$p(o_t = 1|s_t = 1)$: {}".format(round(stats["emp_reg1_sp"], 3)))
        ax[1].plot([], [], ' ', label=r"$p(d_t = 1|s_t = 0)$: {}".format(round(stats["emp_reg0_ap"], 3)))
        ax[1].plot([], [], ' ', label=r"$p(d_t = 1|s_t = 1)$: {}".format(round(stats["emp_reg1_ap"], 3)))
        ax[1].plot([], [], ' ', label=r"Avg Train - $s_t=0$: {}".format(round(stats["avg_train_r0"], 3)))
        ax[1].plot([], [], ' ', label=r"Avg Train - $s_t=1$: {}".format(round(stats["avg_train_r1"], 3)))
        try:
            ax[1].plot([], [], ' ', label="JS-Div. Deviants: {}".format(round(stats["js_div"], 3)))
        except:
            pass
        ax[1].legend(ncol=3, fontsize="small")
        ax[1].set_title("Descriptive Statistics and Train Length Histogram")


        def plot(event, seq_length=seq_length, title=title,
                 sequence=sequence, seq_gen_temp=seq_gen_temp,
                 matlab_out=matlab_out):

            if event.key == 's':
                event.canvas.figure.savefig(fig_dir + title +'.png', dpi=300)
                plt.close()

                save(sequence, seq_gen_temp, matlab_out)

            elif event.key == "n":
                plt.close()
                sequence = seq_gen_temp.sample(seq_length)
                stats, reg_0s, reg_1s = calc_stats(sequence, False)

                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), dpi=300)
                fig.tight_layout()
                coloring = []
                for j in range(len(sequence[:, 0])):
                    if sequence[j, 1] == 0: coloring += "g"
                    elif sequence[j, 1] == 1: coloring += "b"
                    else: coloring += "r"

                ax[0].scatter(np.arange(200), sequence[:200, 2], s=4.5, c=coloring)
                ax[0].set_title("First 200 Trials of Length {} Block".format(sequence.shape[0]))

                ax[1].hist(reg_0s[:, 2], density=True, label=r"Regime 0 ($s_t=0$)", alpha=0.5, range=(0, 20), color="g")
                ax[1].hist(reg_1s[:, 2], density=True, label=r"Regime 1 ($s_t=1$)", alpha=0.5, range=(0, 20), color="b")
                # Add extra info as additional lines with label in legend
                ax[1].plot([], [], ' ', label=r"$p(o_t = 1|s_t = 0)$: {}".format(round(stats["emp_reg0_sp"], 3)))
                ax[1].plot([], [], ' ', label=r"$p(o_t = 1|s_t = 1)$: {}".format(round(stats["emp_reg1_sp"], 3)))
                ax[1].plot([], [], ' ', label=r"$p(d_t = 1|s_t = 0)$: {}".format(round(stats["emp_reg0_ap"], 3)))
                ax[1].plot([], [], ' ', label=r"$p(d_t = 1|s_t = 1)$: {}".format(round(stats["emp_reg1_ap"], 3)))
                ax[1].plot([], [], ' ', label=r"Avg Train - $s_t=0$: {}".format(round(stats["avg_train_r0"], 3)))
                ax[1].plot([], [], ' ', label=r"Avg Train - $s_t=1$: {}".format(round(stats["avg_train_r1"], 3)))
                try:
                    ax[1].plot([], [], ' ', label="JS-Div. Deviants: {}".format(round(stats["js_div"], 3)))
                except:
                    pass
                ax[1].legend(ncol=3, fontsize="small")
                ax[1].set_title("Descriptive Statistics and Train Length Histogram")

                fig.canvas.mpl_connect('key_press_event', plot)
                plt.show()

        fig.canvas.mpl_connect('key_press_event', plot)
        plt.show()
    else:
        save(sequence, seq_gen_temp, matlab_out)




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
    parser.add_argument('-obs_change', '--prob_obs_change', nargs='+',
                        help="Probability of sampling observations",
                        action="store", type=float)
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
    parser.add_argument('-v', '--verbose',
                        action="store_true",
                        default=False,
                        help='Get status printed out')

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
    verbose = args.verbose

    gen_temp = seq_gen(order, prob_catch, prob_regime_init, prob_regime_change,
                       prob_obs_init, prob_obs_change, verbose)

    # sequence = gen_temp.sample(seq_length)
    sample_and_save(gen_temp, seq_length, title, matlab_out, plot_seq)
    """
    pythonw seq_gen.py -t 2nd_5_01_5_10_200 -obs_change 0.45 0.45 0.05 0.05 0.05 0.05 0.45 0.45 -order 2 -matlab
    """
