import numpy as np

def first2dimstolast(matrix):
    """
    Function that swaps the first two dims of a numpy array to the last two dims,
    preserving the order of the dims.
    Useful for swapping the dims of an array containing ABCD matrices.
    """
    return np.moveaxis(matrix,[0,1],[-2,-1])

def last2dimstofirst(matrix):
    """
    Function that swaps the first two dims of a numpy array to the last two dims,
    preserving the order of the dims.
    Useful for swapping the dims of an array containing ABCD matrices.
    """
    return np.moveaxis(matrix,[-2,-1],[0,1])



def lognormal(mean=0.0, sigma=1.0, size=None):
    """
    Lognormal function using a mean and sigma of a normal distribution.

    """
    rng = np.random.default_rng()
    E_x = mean
    VAR_x = sigma**2

    mu_log = np.log(E_x/( np.sqrt( VAR_x / (E_x**2) + 1 ) ))
    sigma_log = np.sqrt( np.log( VAR_x / (E_x**2) + 1 ) )

    return rng.lognormal(mu_log,sigma_log,size)

def res_variance(f0,Ql,Qi,sigma_f0,sigma_Qc):
    if np.isinf(Qi):
        Qc = 2 * Ql
    else:
        Qc = 2 * (Ql * Qi)/(Qi - Ql)
    
    df = f0 / Ql
    
    f0_var = lognormal(f0,df*sigma_f0)
    try:
        Qc_var = lognormal(Qc,Qc*sigma_Qc)
        assert Qc_var > 0, "Qc variance causes <0 Qc value"
    except AssertionError:
        Qc_var = Qc
        raise UserWarning("consider decreasing the variance applied")

    if np.isinf(Qi):
        Ql_var = Qc_var / 2
    else:
        Ql_var = (Qc_var * Qi) / (Qc_var + 2 * Qi)
    
    return f0_var, Ql_var


def ABCD_eye(size,**ndarray_kwargs):
    """
    Create a ABCD matrix compatible array of unity (I) matrices for a given size.
    This results in the 2 by 2 unity array to be in the last two dimensions, which is useful for matrix multiplications.
    """
    size = tuple(size) if np.iterable(size) else (size,)
    return np.tile(np.eye(2,**ndarray_kwargs),reps=size+(1,1))
