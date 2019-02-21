import logging
import numpy as np
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
from utils.helpers import normalize

logger_pymc3 = logging.getLogger('pymc3')
logger_pymc3.setLevel(logging.ERROR)


def run_model_estimation(int_point, y_elec, surprise_reg, model_type):
    """
    Inputs: int_point - sampling point in interstimulus interval
            y_elec - array with eeg recordings (num_trials x num_interstim_rec)
            surprise_reg - num_trials x 1 surprise from Bayesian learning model
            model_type - regression model
    Output: Log model evidence/Negative free energy from VI on Bayesian model
    """
    # Normalize the data and regressor to lie within 0, 1
    y_std = normalize(y_elec[:, int_point])
    surprise_reg_std = normalize(surprise_reg)

    # Select specific model OLS/Hierarchical
    if model_type == "OLS":
        model = OLS_model(y_std, surprise_reg_std)
    elif model_type == "Hierarchical":
        model = hierarchical_model(y_std, surprise_reg_std)
    elif model_type == "Bayesian-MLP":
        model = Bayesian_NN(y_std, surprise_reg_std)
    else:
        raise "Provide a valid model type"

    # Run the Variational Inference scheme with ADVI
    # ADVI - Automatic Differentiation VI
    with model:
        inference = pm.ADVI()
        approx = pm.fit(method=inference,
                        callbacks=[pm.callbacks.CheckParametersConvergence(diff='absolute')],
                        n=30000,
                        progressbar=0)
    # return full optimization trace of free energy
    return -approx.hist[-1]


def OLS_model(y_elec, surprise):
    data = dict(y_elec=y_elec, surprise=surprise)
    with pm.Model() as mdl_ols:
        # Define Normal priors on Params - Ridge
        b0 = pm.Normal('b0', mu=0, sd=100)
        b1 = pm.Normal('b1', mu=0, sd=100)

        # Define Linear model
        y_est = b0 + b1 * data['surprise']

        # Define Normal LH with HalfCauchy noise (fat tails, equiv to HalfT 1DoF)
        sigma_y = pm.HalfCauchy('sigma_y', beta=10)
        likelihood = pm.Normal('likelihood', mu=y_est, sd=sigma_y,
                               observed=data['y_elec'])
    return mdl_ols


def hierarchical_model(y_elec, surprise):
    data = dict(y_elec=y_elec, surprise=surprise)
    with pm.Model() as mdl_hierarchical:
        eta = pm.Normal('eta', 0, 1, shape=y_elec.shape[0])
        mu = pm.Normal('mu', 0, 1e6)
        tau = pm.HalfCauchy('tau', 5)
        theta = pm.Deterministic('theta', mu + tau*eta)

        sigma_y = pm.HalfCauchy('sigma', beta=10)
        likelihood = pm.Normal('likelihood', mu=theta, sd=sigma_y,
                               observed=data['y_elec'])
    return mdl_hierarchical


def Bayesian_NN(y_elec, surprise, n_hidden=32):
    # Initialize random weights between each layer
    init_hidden = np.random.randn(surprise.shape[0], n_hidden).astype(float)
    init_out = np.random.randn(n_hidden).astype(float)

    with pm.Model() as mdl_nn:
        # Weights from input to hidden layer
        weights_in_hidden = pm.Normal('w_in_hidden', 0, sd=1,
                                      shape=(surprise.shape[0], n_hidden),
                                      testval=init_hidden)

        # Weights from hidden layer to output
        weights_hidden_out = pm.Normal('w_hidden_out', 0, sd=1,
                                       shape=(n_hidden,), testval=init_out)

        # Build neural-network using tanh activation function
        act_hidden = pm.math.tanh(pm.math.dot(surprise, weights_in_hidden))
        act_out = pm.math.tanh(pm.math.dot(act_hidden, weights_hidden_out))

        # Linear Regression -> Normal Likelihood with robust Cauchy prior of sd
        sigma_y = pm.HalfCauchy('sigma', beta=10)
        likelihood = pm.Normal('likelihood', mu=act_out, sd=sigma_y,
                               observed=y_elec)
    return mdl_nn
