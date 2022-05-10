import arviz as az
import numpy as np
import pymc3 as pm
from arviz import InferenceData

draws = 500
chains = 2

eight_school_data = {
    "J": 8,
    "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
    "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
}

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sd=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta_tilde = pm.Normal("theta_tilde", mu=0, sd=1, shape=eight_school_data["J"])
    theta = pm.Deterministic("theta", mu + tau * theta_tilde)
    pm.Normal(
        "obs", mu=theta, sd=eight_school_data["sigma"], observed=eight_school_data["y"]
    )

    trace = pm.sample(draws, chains=chains)
    prior = pm.sample_prior_predictive()
    posterior_predictive = pm.sample_posterior_predictive(trace)

    pm_data = az.from_pymc3(
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={"school": np.arange(eight_school_data["J"])},
        dims={"theta": ["school"], "theta_tilde": ["school"]},
    )

joint_log_lik = np.sum(pm_data.log_likelihood.stack(draws=("chain", "draw")).obs.values, axis = 0)

alpha = 0.5
log_lik_ps = (alpha - 1) * joint_log_lik

new_group = {
    "powerscale_weights":
    {"likelihood" :
     log_lik_ps
     }
}


pm_data.add_groups(new_group)

w = pm_data.powerscale_weights.stack(draws = ("chain", "draw")).likelihood.values

mu = pm_data.posterior.stack(draws = ("chain", "draw")).mu.values

base_mean = np.average(mu)
scaled_mean = np.average(mu, weights = w, axis = 0)

