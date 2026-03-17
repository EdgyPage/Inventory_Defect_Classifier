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