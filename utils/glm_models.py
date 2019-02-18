import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence


def normalize(a):
    return (a - np.min(a))/np.ptp(a)


def run_model_estimation(int_point, y_elec, surprise_reg, model_type):
    # Normalize the data and regressor to lie within 0, 1
    y_std = normalize(y_elec[:, int_point])
    surprise_reg_std = normalize(surprise_reg)

    if model_type == "OLS":
        model = OLS_model(y_std, surprise_reg_std)
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
