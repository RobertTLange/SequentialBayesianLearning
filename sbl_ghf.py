import os
import argparse
import numpy as np
from hhmm_seq_gen import hhmm

results_dir = os.getcwd() + "/results/"




if __name__ == "__main__":
    hhmm_temp = hhmm(prob_regime_init, prob_regime_change,
                     prob_obs_init, prob_obs_change)
    hhmm_seq = hhmm_temp.sample_seq(seq_length)[:, [1,2]]
