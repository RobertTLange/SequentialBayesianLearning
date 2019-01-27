import pymc3 as pm


def run_model_estimation(y_elec, surprise,
                         model_type, num_traces):
    data = dict(y_elec=y_elec, surprise=PS)

    if model_type == "OLS":
        model = OLS_model(data, num_traces)

    with neural_network:
        inference = pm.ADVI()
        approx = pm.fit(n=30000, method=inference)
    return


def OLS_model(data, num_traces):
    with pm.Model() as mdl_ols:
        ## define Normal priors to give Ridge regression
        b0 = pm.Normal('b0', mu=0, sd=100)
        b1 = pm.Normal('b1', mu=0, sd=100)

        ## define Linear model
        y_est = b0 + b1 * data['surprise']

        ## Define Normal likelihood with HalfCauchy noise (fat tails, equiv to HalfT 1DoF)
        sigma_y = pm.HalfCauchy('sigma_y', beta=10)
        likelihood = pm.Normal('likelihood', mu=y_est, sd=sigma_y, observed=df_lin['y_elec'])
    return mdl_ols
