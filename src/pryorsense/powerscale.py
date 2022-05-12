import arviz as az
import numpy as np
import pymc3 as pm
from arviz import InferenceData

draws = 2000
chains = 4

data = {
    "y": np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
}

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sd=1)
    sigma = pm.HalfNormal("sigma", sd=2.5)
    pm.Normal(
        "obs", mu=mu, sd=sigma, observed=data["y"]
    )

    trace = pm.sample(draws, chains=chains)
    prior = pm.sample_prior_predictive()
    posterior_predictive = pm.sample_posterior_predictive(trace)

    pm_data = az.from_pymc3(
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive
    )


def powerscale_lw(data, alpha, component):

    if (component == "likelihood"):
        ps = np.sum(pm_data["log_likelihood"].stack(draws = ("chain", "draw")).obs.values, axis = 0)
    elif (component == "prior"):
        ps = pm_data["log_prior"].stack(draws = ("chain", "draw")).values
    
    lw = (alpha - 1) * ps
    
    return lw
