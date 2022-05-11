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

joint_log_lik = np.sum(pm_data.log_likelihood.stack(draws=("chain", "draw")).obs.values, axis = 0)

alpha = 2

new_group = {
    "powerscale":
    {
        "likelihood" : joint_log_lik,
        "prior" : joint_log_lik
    }
}


pm_data.add_groups(new_group)

def powerscale_weights(data, alpha, component):

    if (component is "likelihood"):
        ps = np.sum(pm_data["log_likelihood"].stack(draws = ("chain", "draw")).obs.values, axis = 0)
    elif (component is "prior"):
        ps = pm_data["log_prior"].stack(draws = ("chain", "draw")).values
    
    w = (alpha - 1) * ps
    
    return w

w = powerscale_weights(pm_data, 5, "likelihood")

wps = az.psislw(w)

mu = pm_data.posterior.stack(draws = ("chain", "draw")).mu.values

base_mean = np.average(mu, axis = 0)
scaled_mean = np.average(mu, weights = np.exp(wps), axis = 0)

np.dot(mu, w)

base_sd = np.sqrt(np.var(mu, axis = 0))
scaled_sd = np.sqrt(np.average((mu-scaled_mean)**2, weights=np.exp(wps), axis = 0))
