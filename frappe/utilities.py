
###### Utiities #########

from astroquery.linelists.cdms import CDMS
import jax
import jax.numpy as jnp
import numpy as np
from .constants import *
from astropy import units as u
import warnings
import numpy as np

import jax.numpy as jnp


from scipy.ndimage import gaussian_filter1d


from scipy.interpolate import interp1d as scipy_interp1d

def check_range(x, lower, upper, alpha=10.0):
    # Apply exponential penalty if x is outside [lower, upper]
    penalty_lower = -jnp.exp(-alpha * (x - lower))
    penalty_upper = -jnp.exp(alpha * (x - upper))
    penalty = jnp.where(x < lower, penalty_lower,
                jnp.where(x > upper, penalty_upper, 0.0))
    return penalty

'''
def check_range(x, lower, upper, alpha=100.0):
    """
    x, lower, upper が同じ形状の配列であることを想定。
    境界を越えた各要素に対して、二乗ペナルティを計算します。
    """
    # x < lower の場合、(x - lower) は負の値。その分を二乗してペナルティ。
    # x > upper の場合、(x - upper) は正の値。その分を二乗してペナルティ。
    
    # 配列の各要素に対して、はみ出した量（distance）を計算
    dist_lower = jnp.where(x < lower, x - lower, 0.0)
    dist_upper = jnp.where(x > upper, x - upper, 0.0)
    
    # 合計のペナルティ（各要素ごとの負の値の配列を返す）
    # alpha * 0.5 * dist^2 は、境界で勾配が0から始まり、離れるほど強く戻る力になる
    penalty = -0.5 * alpha * (jnp.square(dist_lower) + jnp.square(dist_upper))
    return penalty
'''


def hankel_transform_0_jax(f, r, k, bessel):
    '''
    Perform the Hankel transform of order 0 using JAX.
    f: jnp.ndarray, function values at radial distances r (shape: [n_r])
    r: jnp.ndarray, radial distances (shape: [n_r])
    k: jnp.ndarray, spatial frequencies (shape: [n_k])
    bessel: jnp.ndarray, precomputed Bessel function values (shape: [n_k, n_r])
    Returns the Hankel transform values at spatial frequencies k (shape: [n_k]).
    '''
    
    dr = jnp.gradient(r)
    fr = f * r

    #def integrate(ki):
    #    integrand = fr * j0(k * r) 
    #    return jnp.sum(integrand * dr)
    
    return jnp.sum( 2*np.pi * fr * bessel * dr, axis=1)

def rbf_kernel(X1, X2, variance, lengthscale):
    '''
    Compute the Radial Basis Function (RBF) kernel between two sets of input points using JAX.
    X1: jnp.ndarray, first set of input points (shape: [n1, d])
    X2: jnp.ndarray, second set of input points (shape: [n2 , d])
    variance: float, variance parameter of the RBF kernel
    lengthscale: float, lengthscale parameter of the RBF kernel
    Returns the RBF kernel matrix (shape: [n1, n2]).
    '''
    
    sq_dist = jnp.sum(X1**2, 1)[:, None] + jnp.sum(X2**2, 1)[None, :] - 2 * jnp.dot(X1, X2.T)
    
    return variance**2 * jnp.exp(-0.5 / lengthscale**2 * sq_dist)

def B(nu, T):
    '''
    Calculate the Planck function B(nu, T).
    nu: jnp.ndarray or float, frequency in Hz
    T: jnp.ndarray or float, temperature in Kelvin
    Returns the Planck function values.
    ''' 

    return 2*h*nu**3/c**2 / ( jnp.exp(h*nu/k_B/T) - 1 )
    
def sigmoid_transform(x, min_val=0.0, max_val=1.0, leak = 0.01):
    '''
    Apply a sigmoid transformation to the input array x.
    The transformed values will be in the range [min_val, max_val].
    x: jnp.ndarray, input array to be transformed
    min_val: float, minimum value of the transformed output
    max_val: float, maximum value of the transformed output
    Returns the transformed array with values in the range [min_val, max_val].
    '''
    
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))

def sigmoid_transform_old(x, min_val=0.0, max_val=1.0):
    y = (2.0 / jnp.pi) * jnp.arctan(x)   # 範囲は (-1, 1)
    y01 = 0.5 * (y + 1.0)
    return min_val + (max_val - min_val) * y01


def F(tau, omega):
    '''
    Calculate the function F(tau, omega) used in radiative transfer.
    tau: jnp.ndarray or float, optical depth
    omega: jnp.ndarray or float, single scattering albedo
    Returns the computed values of F(tau, omega).
    Ref. Miyake & Nakagawa 1993, Icarus, 106, 20; Sierra et al. 2020, ApJ, 892, 136
    '''
    
    w = omega
    
    term1 = (jnp.sqrt(1 - w) - 1.0) * jnp.exp(-jnp.sqrt(3.0 / (1.0 - w)) * tau)
    
    A_num = 1.0 - jnp.exp(-(jnp.sqrt(3.0 * (1.0 - w)) + 1.0) * tau / (1.0 - w))
    A_den = jnp.sqrt(3.0 * (1.0 - w)) + 1.0
    A = A_num / A_den
    
    B_num = jnp.exp(-tau / (1.0 - w)) - jnp.exp(-jnp.sqrt(3.0 / (1.0 - w)) * tau)
    B_den = jnp.sqrt(3.0 * (1.0 - w)) - 1.0
    B = B_num / B_den
    
    term2 = (jnp.sqrt(1 - w) + 1.0)
    
    denom = term1 - term2
    
    return  (A + B) / denom


def f_I(nu, incl, T, Sigma_d, k_abs_tot, k_sca_eff_tot):
    '''
    Calculate the intensity I(nu) using radiative transfer with scattering.
    nu: jnp.ndarray or float, frequency in Hz
    incl: jnp.ndarray or float, inclination angle in radians
    T: jnp.ndarray or float, temperature in Kelvin
    Sigma_d: jnp.ndarray or float, dust surface density in g/cm^2
    dust_params: list of jnp.ndarray or float, dust parameters (e.g., maximum grain size). Assuming the order matches the interpolators.
    f_log10_ka: function, interpolator for log10 of absorption opacity
    f_log10_ks: function, interpolator for log10 of scattering opacity
    Returns the computed intensity I(nu).
    ''' 

    ka = k_abs_tot
    ks = k_sca_eff_tot

    chi = ka + ks
    omega = ks / chi
    
    tau = ka * Sigma_d / jnp.cos(incl)

    return B(nu, T) * (  1 - jnp.exp( -tau/(1-omega) ) + omega*F(tau, omega)  )


def size_average_opacity( log10_a, log10_k_abs, log10_k_sca_eff, log10_a_max, log10_a_min, q):

    a = 10**log10_a
    a_max = 10**log10_a_max
    a_min = 10**log10_a_min

    mask = jnp.where( (a <= a_max) & (a >= a_min), 1.0, 0.0)
    n = a**(4.0-q) * mask #* jnp.exp(-(a / a_max)**gamma)  * jnp.exp(-(a_min/a)**gamma)

    sum_n = jnp.sum(n)
    
    log10_k_abs_tot = jnp.log10(jnp.dot(n, 10**log10_k_abs) /sum_n)
    log10_k_sca_eff_tot = jnp.log10(jnp.dot(n, 10**log10_k_sca_eff) /sum_n)

    return log10_k_abs_tot, log10_k_sca_eff_tot


def create_opacity_table(lam, a, k_abs, k_sca_eff, lam0, log10_a_dense, q_dense, smooth=True, log10_a_smooth=0.1, log10_a_min = -5.0):
    '''
    Create opacity tables by interpolating and averaging over grain size distributions.
    lam: np.ndarray, wavelengths in microns (shape: [M])
    a: np.ndarray, grain sizes in microns (shape: [N])
    k_abs: np.ndarray, absorption opacities (shape: [N, M])
    k_sca_eff: np.ndarray, effective scattering opacities (shape: [N, M])
    lam0: np.ndarray, target wavelengths for interpolation in microns (shape: [L])
    log10_a_dense: jnp.ndarray, dense grid of log10 grain sizes in microns (shape: [P])
    q_dense: jnp.ndarray, dense grid of size distribution indices (shape: [Q])
    smooth: bool, whether to apply smoothing to the interpolated opacities
    log10_a_smooth: float, smoothing scale in log10 grain size
    Returns:
    log10_k_abs_tot: jnp.ndarray, size-averaged log10 absorption opacities (shape: [P, Q])
    log10_k_sca_eff_tot: jnp.ndarray, size-averaged log10 effective scattering opacities (shape: [P, Q])
    ''' 

    log10_k_abs_itp_lam = scipy_interp1d( np.log10(lam), np.log10(k_abs), axis=1, kind="cubic")( np.log10(lam0) )
    log10_k_sca_eff_itp_lam = scipy_interp1d( np.log10(lam), np.log10(k_sca_eff), axis=1, kind="cubic")( np.log10(lam0) )

    log10_k_abs_itp = jnp.array( scipy_interp1d( np.log10(a), log10_k_abs_itp_lam , kind='cubic', bounds_error=False, fill_value="extrapolate")(log10_a_dense))
    log10_k_sca_eff_itp = jnp.array( scipy_interp1d( np.log10(a), log10_k_sca_eff_itp_lam, kind='cubic', bounds_error=False, fill_value="extrapolate")(log10_a_dense))
                
    # smoothing to avoid Mie interference wiggles

    if smooth:
        sigma_a = log10_a_smooth/ ( log10_a_dense[1] - log10_a_dense[0])
                
        log10_k_abs_itp  = jnp.array( gaussian_filter1d( np.array(log10_k_abs_itp), sigma=sigma_a ) )
        log10_k_sca_eff_itp = jnp.array( gaussian_filter1d( np.array(log10_k_sca_eff_itp), sigma=sigma_a ) )



                
        vmap_over_q = jax.vmap(
                    size_average_opacity,
                    in_axes=(None, None, None, None, None, 0)  # q だけが配列
                )

               
        vmap_over_a_and_q = jax.vmap(
                    vmap_over_q,
                    in_axes=(None, None, None, 0, None, None)  # log10_a_max だけが配列
                )

                
        log10_k_abs_tot, log10_k_sca_eff_tot = vmap_over_a_and_q(
                    log10_a_dense,
                    log10_k_abs_itp,
                    log10_k_sca_eff_itp,
                    log10_a_dense,
                    log10_a_min,
                    q_dense
                )
        
        return log10_k_abs_tot, log10_k_sca_eff_tot