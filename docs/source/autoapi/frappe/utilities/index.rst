frappe.utilities
================

.. py:module:: frappe.utilities


Functions
---------

.. autoapisummary::

   frappe.utilities.check_range
   frappe.utilities.hankel_transform_0_jax
   frappe.utilities.rbf_kernel
   frappe.utilities.B
   frappe.utilities.sigmoid_transform
   frappe.utilities.sigmoid_transform_old
   frappe.utilities.F
   frappe.utilities.f_I
   frappe.utilities.size_average_opacity
   frappe.utilities.create_opacity_table


Module Contents
---------------

.. py:function:: check_range(x, lower, upper, alpha=10.0)

.. py:function:: hankel_transform_0_jax(f, r, k, bessel)

   Perform the Hankel transform of order 0 using JAX.
   f: jnp.ndarray, function values at radial distances r (shape: [n_r])
   r: jnp.ndarray, radial distances (shape: [n_r])
   k: jnp.ndarray, spatial frequencies (shape: [n_k])
   bessel: jnp.ndarray, precomputed Bessel function values (shape: [n_k, n_r])
   Returns the Hankel transform values at spatial frequencies k (shape: [n_k]).


.. py:function:: rbf_kernel(X1, X2, variance, lengthscale)

   Compute the Radial Basis Function (RBF) kernel between two sets of input points using JAX.
   X1: jnp.ndarray, first set of input points (shape: [n1, d])
   X2: jnp.ndarray, second set of input points (shape: [n2 , d])
   variance: float, variance parameter of the RBF kernel
   lengthscale: float, lengthscale parameter of the RBF kernel
   Returns the RBF kernel matrix (shape: [n1, n2]).


.. py:function:: B(nu, T)

   Calculate the Planck function B(nu, T).
   nu: jnp.ndarray or float, frequency in Hz
   T: jnp.ndarray or float, temperature in Kelvin
   Returns the Planck function values.


.. py:function:: sigmoid_transform(x, min_val=0.0, max_val=1.0, leak=0.01)

   Apply a sigmoid transformation to the input array x.
   The transformed values will be in the range [min_val, max_val].
   x: jnp.ndarray, input array to be transformed
   min_val: float, minimum value of the transformed output
   max_val: float, maximum value of the transformed output
   Returns the transformed array with values in the range [min_val, max_val].


.. py:function:: sigmoid_transform_old(x, min_val=0.0, max_val=1.0)

.. py:function:: F(tau, omega)

   Calculate the function F(tau, omega) used in radiative transfer.
   tau: jnp.ndarray or float, optical depth
   omega: jnp.ndarray or float, single scattering albedo
   Returns the computed values of F(tau, omega).
   Ref. Miyake & Nakagawa 1993, Icarus, 106, 20; Sierra et al. 2020, ApJ, 892, 136


.. py:function:: f_I(nu, incl, T, Sigma_d, k_abs_tot, k_sca_eff_tot)

   Calculate the intensity I(nu) using radiative transfer with scattering.
   nu: jnp.ndarray or float, frequency in Hz
   incl: jnp.ndarray or float, inclination angle in radians
   T: jnp.ndarray or float, temperature in Kelvin
   Sigma_d: jnp.ndarray or float, dust surface density in g/cm^2
   dust_params: list of jnp.ndarray or float, dust parameters (e.g., maximum grain size). Assuming the order matches the interpolators.
   f_log10_ka: function, interpolator for log10 of absorption opacity
   f_log10_ks: function, interpolator for log10 of scattering opacity
   Returns the computed intensity I(nu).


.. py:function:: size_average_opacity(log10_a, log10_k_abs, log10_k_sca_eff, log10_a_max, log10_a_min, q)

.. py:function:: create_opacity_table(lam, a, k_abs, k_sca_eff, lam0, log10_a_dense, q_dense, smooth=True, log10_a_smooth=0.1, log10_a_min=-5.0)

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


