import os
import time
import argparse
import numpy as np

#
from utils.helpers import *
from utils.glm_models import *

# Import relevant SBL surprise regressor modules
import sbl_agents.sbl_cat_dir as sbl_cd
import sbl_agents.sbl_hmm as sbl_hmm

# Set random seed for replicability and define directories
np.random.seed(seed=1234)
results_dir = os.getcwd() + "/results/"

# General Model Settings
model_types = ["SP", "AP", "TP"]
save_results = True
verbose = True

sample_files = [["sub-01/sub-01_ses-1_run-1", "sub-01/sub-01_ses-1_run-2",
                 "sub-01/sub-01_ses-1_run-3", "sub-01/sub-01_ses-1_run-4",
                 "sub-01/sub-01_ses-1_run-5"],
                ["sub-02/sub-02_ses-1_run-1", "sub-02/sub-02_ses-1_run-2",
                 "sub-02/sub-02_ses-1_run-3", "sub-02/sub-02_ses-1_run-4",
                 "sub-02/sub-02_ses-1_run-5"],
                ["sub-04/sub-04_ses-1_run-1", "sub-04/sub-04_ses-1_run-2",
                 "sub-04/sub-04_ses-1_run-3", "sub-04/sub-04_ses-1_run-4",
                 "sub-04/sub-04_ses-1_run-5"]]

eeg_files = ["sub-01/sub-01_sbl"]
subject_list = range(len(eeg_files))

# Select block and electrode for analysis
sampling_rate = 0.3
inter_stim_interval = np.array([-0.05, 0.65])
template = "Subject {} | Block {} | Electrode {} | Reg: {} | Time: {:.2f}"


def main(subject_id, eoi, inter_stim_interval, sampling_rate,
         verbose, results_dir=results_dir):
    if verbose:
        print("Start running T-b-T Analysis for subject {}".format(subject_id))
        print("Interstimulus Interval: {}ms".format(inter_stim_interval))
        print("Electrodes of Interest: {}".format(list(eoi.keys())))
        print("Sampling Rate: {}Hz".format(512*sampling_rate))
    # Create Logging object
    log = ExperimentLog(subject_id=0, num_blocks=5,
                        elec_of_interest=eoi,
                        save_fname=results_dir + "subject_" + str(subject_id) + ".hdf5")

    # Load in the EEG data
    eeg_data = sio.loadmat("data/" + eeg_files[subject_id] + ".mat")

    # Loop over BLOCKS
    for block_id in range(len(sample_files[subject_id])):
        # Load in the specific trial/stimuli sequence
        sample, meta = load_obj("data/" + sample_files[subject_id][block_id] + ".mat")
        seq, hidden = sample[:, 2], sample[:, 1]

        # Compute Surprise Regressors
        CD_PS_SP, CD_BS_SP, CD_CS_SP = sbl_cd.main(seq, hidden, tau=0,
                                                   model_type="SP")
        CD_PS_AP, CD_BS_AP, CD_CS_AP = sbl_cd.main(seq, hidden, tau=0,
                                                   model_type="AP")
        CD_PS_TP, CD_BS_TP, CD_CS_TP = sbl_cd.main(seq, hidden, tau=0,
                                                   model_type="TP")

        # Define dictionary of regressors with which to run analysis
        regressors = {"CD_PS_SP": CD_PS_SP, "CD_BS_SP": CD_BS_SP,
                      "CD_PS_AP": CD_PS_AP, "CD_BS_AP": CD_BS_AP,
                      "CD_PS_TP": CD_PS_TP, "CD_BS_TP": CD_BS_TP}

        # Loop over ELECTRODES OF INTEREST
        for elec_name, elec_id in eoi.items():
            # Get the block- and electrode-specific eeg data
            y_elec, y_tw = get_electrode_data(eeg_data, block_id, elec_id,
                                              inter_stim_interval, sampling_rate,
                                              verbose=False)
            # Get null model once for a block
            # (PS-AP as filler - parallelize does not work with None regressor)
            start = time.time()
            null_model_lme = parallelize_over_samples(y_elec,
                                                      regressor=regressors["CD_PS_AP"],
                                                      reg_model_type="Null")
            t_time = time.time() - start

            log.dump_data(subject_id, block_id, elec_name,
                          "Null", null_model_lme)
            log.dump_data(subject_id, block_id, elec_name,
                          "Sample_Points", y_tw)
            print(template.format(subject_id+1, block_id+1,
                                  elec_name, "Null", t_time))

            # Loop over Different SURPRISE REGRESSORS
            for regressor_type, regressor in regressors.items():

                start = time.time()
                results = parallelize_over_samples(y_elec, regressor,
                                                   reg_model_type)
                t_time = time.time() - start

                log.dump_data(subject_id, block_id,
                              elec_name, regressor_type, results)
                print(template.format(subject_id+1, block_id+1,
                                      elec_name, regressor_type, t_time))


if __name__ == "__main__":
    # Set params for proto-typing
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg_model', '--reg_model_type', action="store",
                        default="OLS", type=str,
                        help="Bayesian Regression Model to run")
    parser.add_argument('-s_id', '--subject_id', action="store",
                        default=0.3, type=int,
                        help="Percent of sampling points to fit")
    parser.add_argument('-s_rate', '--sampling_rate', action="store",
                        default=0.3, type=float,
                        help="Percent of sampling points to fit")
    parser.add_argument('-int_start', '--inter_stim_start', action="store",
                        default=-0.1, type=float,
                        help="Beginning of Interstimulus Interval")
    parser.add_argument('-int_stop', '--inter_stim_stop', action="store",
                        default=0.65, type=float,
                        help="End of Interstimulus Interval")
    # TODO: Somehow add eoi and regressors
    # parser.add_argument('-obs_change', '--prob_obs_change', nargs='+',
    #                     help="Probability of sampling observations",
    #                     action="store", type=float)
    parser.add_argument('-v', '--verbose',
                        action="store_true",
                        default=False,
                        help='Get status printed out')

    args = parser.parse_args()

    elec_of_interest = {"FCz": 47, "FC2": 46, "FC4": 45,
                        "Cz": 48, "C2": 49, "C4": 50,
                        "C6": 51, "CPz": 32, "CP2": 56,
                        "CP4": 55, "CP6": 54}
    eoi = {"Cz": 48, "C2": 49, "C4": 50}

    reg_model_type = args.reg_model_type
    sampling_rate = args.sampling_rate
    inter_stim_interval = np.array([args.inter_stim_start,
                                    args.inter_stim_stop])
    verbose = args.verbose

    subject_id = args.subject_id
    main(subject_id, eoi, inter_stim_interval, sampling_rate, verbose)
