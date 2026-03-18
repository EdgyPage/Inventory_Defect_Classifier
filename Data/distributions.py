import numpy as np

def sample_negative_binomial(n, p, size=1):
    """
    Sample from a negative binomial distribution.
    
    Args:
        n: Number of successes
        p: Probability of success
        size: Number of samples
    
    Returns:
        Sampled values from negative binomial distribution
    """
    return np.random.negative_binomial(n, p, size)


def sample_normal(mean, std, size=1):
    """
    Sample from a normal distribution.
    
    Args:
        mean: Mean of the distribution
        std: Standard deviation
        size: Number of samples
    
    Returns:
        Sampled values from normal distribution
    """
    return np.random.normal(mean, std, size)


def sample_poisson(lam, size=1):
    """
    Sample from a Poisson distribution.
    
    Args:
        lam: Lambda (expected value)
        size: Number of samples
    
    Returns:
        Sampled values from Poisson distribution
    """
    return np.random.poisson(lam, size)


def sample_exponential_decay(lam, size=1):
    """
    Sample from an exponential distribution (exponential decay).
    
    Args:
        scale: Scale parameter (1/lambda)
        size: Number of samples
    
    Returns:
        Sampled values from exponential distribution
    """
    return np.random.exponential(1/lam, size)

def sample_beta(alpha, beta, size=1):
    """
    Draw `size` samples from a Beta(alpha, beta) distribution on [0,1].

    Parameters
    ----------
    alpha : float
        Shape parameter α > 0 (pulls mass toward 1 when > β).
    beta : float
        Shape parameter β > 0 (pulls mass toward 0 when > α).
    size : int or tuple, optional
        Number/shape of samples. Default is 1.

    Returns
    -------
    samples : ndarray
        Array of samples in [0,1].
    """ 
    return np.random.beta(alpha, beta, size=size)

def beta_from_mean_var(mu, var):
    """
    Return alpha, beta for Beta(α,β) with given mean μ and variance var.
    
    var must be < μ(1-μ).
    """
    if var >= mu * (1 - mu):
        raise ValueError("var must be < mu*(1-mu)")
    alpha = mu * (mu * (1 - mu) / var - 1)
    beta_param = alpha * (1 - mu) / mu
    return alpha, beta_param