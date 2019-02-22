import os
import pickle
from scipy import log, log2, array, zeros
import scipy.io as sio
from scipy.special import gamma, digamma, gammaln
from scipy.stats import dirichlet
import numpy as np

results_dir = os.getcwd() + "/results/"


def normalize(a):
    return (a - np.min(a))/np.ptp(a)


def save_obj(obj, title):
    with open(title + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(title):
    filename, file_extension = os.path.splitext(title)
    if file_extension == ".mat":
        out = sio.loadmat(title)
        sample = out["C"][0][0][5][0][0][0]
        meta = {}
        meta["prob_obs_init"] = out["C"][0][0][5][0][0][1]
        meta["prob_regime_init"] = out["C"][0][0][5][0][0][2]
        meta["prob_obs_change"] = out["C"][0][0][5][0][0][3]
        meta["prob_regime_change"] = out["C"][0][0][5][0][0][4]
        return sample, meta
    else:
        with open(title, 'rb') as f:
            return pickle.load(f)


def kl_general(p, q):
    """Compute the KL divergence between two discrete probability distributions
    The calculation is done directly using the Kullback-Leibler divergence,
    KL( p || q ) = sum_{x} p(x) ln( p(x) / q(x) )
    Natural logarithm is used!
    """
    if (p == 0.).sum() + (q == 0.).sum() > 0:
        raise "Zero bins found"
    return (p*(np.log(p) - np.log(q))).sum()


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


def get_electrode_data(eeg_data, block_id, elec_id,
                       inter_stim_interval=np.array([-0.05, 0.65]),
                       percent_resolution=1, verbose=True):
    num_blocks = 5
    num_trials = 4000
    sampling_rate = 512
    num_inter_stim_rec = int(sampling_rate *
                             (inter_stim_interval[1] - inter_stim_interval[0]))  # round down!
    # Subselect eeg + recording timestamps from raw data object in .mat
    """
    Structure of eeg_raw/eeg_times obj: Sampling rate of 512 points per sec
        - raw: # rows = number of blocks, # cols = # of electrodes (see EOI)
        - times: # rows = # of trials and start of blocks (last rows)
    """
    eeg_raw = eeg_data["data"][0]
    eeg_time = eeg_data["data"][1]
    # Select data according to block and electrode id
    elec_bl_raw = eeg_raw[block_id][elec_id]
    eeg_bl_time = eeg_time[block_id].flatten()
    # Select block-specific event times from from raw data in .mat file
    """
    Structure of event_times object: Rows 1-4000: Events/Trials
        First Col: Boolean for Bad Quality Trial
        Second Col: Form of stimulus/trial see trial_coding_lookup object
        Third Col: Time of trial - use to match with elec_bl_raw to get data

    block_start_times: Final rows of event_times yield the starting times of blocks
        - use to subselect specific data with the help of the trial times
    """
    event_times = eeg_data["event_times"][0]
    event_times = np.array(event_times.tolist()).reshape((num_trials+num_blocks, 3))
    block_start_times = []

    for i in range(len(event_times[num_trials:])):
        block_start_times.append(event_times[num_trials:][i][2])
    # Append final point in time and sanity check
    block_start_times.append(event_times[num_trials-1][2])
    if len(block_start_times) != (num_blocks + 1):
        raise "Something is wrong with data shape: Wrong number of blocks!"

    # Time interval of specific blocks - and id of events in block
    time_int = block_start_times[block_id:block_id+2]
    start_idx_block = np.where(event_times[:, 2] > time_int[0])
    stop_idx_block = np.where(event_times[:, 2] < time_int[1])
    block_event_idx = np.intersect1d(start_idx_block, stop_idx_block)

    # Select event times based on start/stop of block
    events_in_block = event_times[block_event_idx, 2]
    if len(events_in_block) != (num_trials/num_blocks):
        raise "Something is wrong with data shape: Wrong number of events!"

    events_int_start = events_in_block + inter_stim_interval[0]
    events_int_stop = events_in_block + inter_stim_interval[1]

    # Loop over all events and append an array which corresponds to all sampled
    # points within the interstimulus interval
    eeg_data_out = np.empty((0, num_inter_stim_rec), float)

    for t in range(events_in_block.shape[0]):
        start_idx_stim = np.where(eeg_bl_time >= events_int_start[t])
        stop_idx_stim = np.where(eeg_bl_time <= events_int_stop[t])
        stim_event_idx = np.intersect1d(start_idx_stim,
                                        stop_idx_stim)[:num_inter_stim_rec]
        eeg_temp_array = elec_bl_raw[stim_event_idx]
        eeg_data_out = np.vstack((eeg_data_out, eeg_temp_array))

    # Peform downsampling for desired resolution - pick from lin space
    desired_samples = int(percent_resolution*num_inter_stim_rec)
    sample_idx = np.linspace(0, num_inter_stim_rec-1, desired_samples, dtype=int)
    sample_time_window = np.linspace(inter_stim_interval[0],
                                     inter_stim_interval[1],
                                     desired_samples)

    if verbose:
        print("Done selecting block/electrode specific data for [{}, {}]ms int".format(inter_stim_interval[0], inter_stim_interval[1]))
        if percent_resolution != 1:
            print("Downsampled original {} Hz Sampling Rate to {} Hz.". format(sampling_rate, int(desired_samples/(inter_stim_interval[1] - inter_stim_interval[0]))))

    # return the eeg array subselected for block, time window, and sampling
    # return the exact time points in sample_time_window array
    return eeg_data_out[:, sample_idx], sample_time_window
