import numpy as np
import pandas as pd
from scipy.special import expit  # logistic / inverse logit

n = 10000
rng = np.random.default_rng(42)

# ----- 1. Global Defect Rate (true + error) -----
# True per-SKU/global rate ~ Beta
global_defect_true = rng.beta(a=2, b=200, size=n)  # mean ~ 0.01

# Add normal noise, clamp to [0,1]
eps_g = rng.normal(0, 0.005, size=n)
global_defect_obs = np.clip(global_defect_true + eps_g, 0, 1)

# ----- 2. abs(current - mean volume) -----
vol_true = rng.normal(loc=0, scale=1.5, size=n)  # standardized deviation
vol_true = np.abs(vol_true)

eps_v = rng.normal(0, 0.2, size=n)
vol_obs = np.maximum(0, vol_true + eps_v)

# ----- 3. Storage Config (categorical with misclassification) -----
storage_levels = np.array(["XS", "S", "M", "L"])
p_levels = np.array([0.1, 0.4, 0.3, 0.2])
storage_true = rng.choice(storage_levels, size=n, p=p_levels)

# Simple misclassification: 5% chance to flip to another random level
flip = rng.random(n) < 0.05
storage_obs = storage_true.copy()
storage_obs[flip] = rng.choice(storage_levels, size=flip.sum())

# One-hot encode for modeling
storage_dummies = pd.get_dummies(storage_obs, prefix="Storage", drop_first=True)

# ----- 4. Aisle Hold Percent (Beta + binomial noise) -----
aisle_hold_true = rng.beta(a=1.5, b=30, size=n)
N_h = 50  # sample size for estimate
K_h = rng.binomial(N_h, aisle_hold_true)
aisle_hold_obs = K_h / N_h

# ----- 5. Num pick events / in clique (NB + discrete noise) -----
# Negative binomial parameterization via mean and dispersion
def rnbinom(mu, k, size):
    p = k / (k + mu)
    return rng.negative_binomial(k, p, size=size)

pick_events_true = rnbinom(mu=5, k=2, size=n)
pick_events_clique_true = rnbinom(mu=8, k=2, size=n)

# Add discrete normal-ish noise, truncate at 0
eps_pe = np.round(rng.normal(0, 1, size=n)).astype(int)
eps_pec = np.round(rng.normal(0, 1, size=n)).astype(int)

pick_events_obs = np.maximum(0, pick_events_true + eps_pe)
pick_events_clique_obs = np.maximum(0, pick_events_clique_true + eps_pec)

# ----- 6. Num picks / picks in clique (similar) -----
picks_true = rnbinom(mu=20, k=3, size=n)
picks_clique_true = rnbinom(mu=30, k=3, size=n)

eps_pk = np.round(rng.normal(0, 2, size=n)).astype(int)
eps_pkc = np.round(rng.normal(0, 2, size=n)).astype(int)

picks_obs = np.maximum(0, picks_true + eps_pk)
picks_clique_obs = np.maximum(0, picks_clique_true + eps_pkc)

# ----- 7. Defect in related receive (binary with misclassification) -----
z_true = rng.binomial(1, 0.05, size=n)
alpha_fp = 0.01  # false positive
beta_fn = 0.10   # false negative

z_obs = z_true.copy()
# false positives
fp = (z_true == 0) & (rng.random(n) < alpha_fp)
# false negatives
fn = (z_true == 1) & (rng.random(n) < beta_fn)
z_obs[fp] = 1
z_obs[fn] = 0

# ----- 8. Time in location (Gamma + noise) -----
time_true = rng.gamma(shape=2.0, scale=3.0, size=n)  # mean ~ 6 days
eps_t = rng.normal(0, 0.5, size=n)
time_obs = np.maximum(0, time_true + eps_t)
time_obs = np.round(time_obs)

# ----- 9. Current max volume (lognormal + multiplicative error) -----
max_vol_true = rng.lognormal(mean=3.0, sigma=0.5, size=n)
u_err = rng.lognormal(mean=0.0, sigma=0.1, size=n)
max_vol_obs = max_vol_true * u_err

# ----- Assemble observed design matrix -----
X = pd.DataFrame({
    "GlobalDefectRate": global_defect_obs,
    "AbsVolDeviation": vol_obs,
    "AisleHoldPct": aisle_hold_obs,
    "NumPickEvents": pick_events_obs,
    "NumPickEventsClique": pick_events_clique_obs,
    "NumPicks": picks_obs,
    "NumPicksClique": picks_clique_obs,
    "DefectInRelatedReceive": z_obs,
    "TimeInLocation": time_obs,
    "CurrentMaxVolume": max_vol_obs
})

X = pd.concat([X, storage_dummies], axis=1)

# ----- Define a logistic model and simulate defects -----
# example coefficients (choose whatever structure you want)
beta = np.array([
    -3.0,   # intercept
    5.0,    # GlobalDefectRate
    0.2,    # AbsVolDeviation
    1.5,    # AisleHoldPct
    0.03,   # NumPickEvents
    0.02,   # NumPickEventsClique
    0.01,   # NumPicks
    0.01,   # NumPicksClique
    1.0,    # DefectInRelatedReceive
    0.02,   # TimeInLocation
    -0.0005 # CurrentMaxVolume
])

# add coefficients for storage dummies (XS baseline)
storage_cols = [c for c in X.columns if c.startswith("Storage_")]
beta_storage = np.array([0.3] * len(storage_cols))  # same effect for illustration

beta_full = np.concatenate([beta, beta_storage])

# design matrix as numpy
X_mat = np.column_stack([
    np.ones(n),  # intercept
    X[[
        "GlobalDefectRate", "AbsVolDeviation", "AisleHoldPct",
        "NumPickEvents", "NumPickEventsClique",
        "NumPicks", "NumPicksClique",
        "DefectInRelatedReceive", "TimeInLocation",
        "CurrentMaxVolume"
    ]].values,
    X[storage_cols].values
])

eta = X_mat @ beta_full
p_defect = expit(eta)  # probability in [0,1]

# simulate observed defect indicator
D = rng.binomial(1, p_defect)

sim_data = X.copy()
sim_data["Defect"] = D


