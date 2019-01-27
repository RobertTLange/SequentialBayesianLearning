import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence


def run_model_estimation(y_elec, surprise, model_type):
    data = dict(y_elec=y_elec, surprise=surprise)

    if model_type == "OLS":
        model = OLS_model(y_elec, surprise)
    else:
        raise "Provide a valid model type"

    with model:
        inference = pm.ADVI()
        approx = pm.fit(method=inference,
                        callbacks=[pm.callbacks.CheckParametersConvergence(diff='absolute')],
                        n=30000,
                        progressbar=0)
    # return full optimization trace of free energy
    return -approx.hist


def OLS_model(y_elec, surprise):
    data = dict(y_elec=y_elec, surprise=surprise)
    with pm.Model() as mdl_ols:
        # Define Normal priors to give Ridge regression
        b0 = pm.Normal('b0', mu=0, sd=100)
        b1 = pm.Normal('b1', mu=0, sd=100)

        # Define Linear model
        y_est = b0 + b1 * data['surprise']

        # Define Normal likelihood with HalfCauchy noise (fat tails, equiv to HalfT 1DoF)
        sigma_y = pm.HalfCauchy('sigma_y', beta=10)
        likelihood = pm.Normal('likelihood', mu=y_est, sd=sigma_y, observed=data['y_elec'])
    return mdl_ols


def Hierarchical_model(y_elec, surprise):
    data = dict(y_elec=y_elec, surprise=surprise)

    with pm.Model() as mdl_hierarchical:
        eta = pm.Normal('eta', 0, 1, shape=y_elec.shape[0])
        mu = pm.Normal('mu', 0, 1e6)
        tau = pm.HalfCauchy('tau', 5)

        theta = pm.Deterministic('theta', mu + tau*eta)

        obs = pm.Normal('obs', theta, sigma, observed=y)
    return mdl_hierarchical
