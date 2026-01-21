frappe.model
============

.. py:module:: frappe.model


Attributes
----------

.. autoapisummary::

   frappe.model.c
   frappe.model.k_B
   frappe.model.h
   frappe.model.au
   frappe.model.G
   frappe.model.M_sun
   frappe.model.m_p


Classes
-------

.. autoapisummary::

   frappe.model.model
   frappe.model.observation
   frappe.model.inference
   frappe.model.results
   frappe.model.plot


Module Contents
---------------

.. py:data:: c

.. py:data:: k_B

.. py:data:: h

.. py:data:: au

.. py:data:: G

.. py:data:: M_sun

.. py:data:: m_p

.. py:class:: model(incl, r_out, N_GP, userdef_vis_model=None, flux_uncert=True, jitter=1e-06, hyperparameters_fixed=True)

   class for FRAPPE model

   .. attribute:: set_parameters

      method to set model parameters

   .. attribute:: set_observations

      method to set observational data

   .. attribute:: set_opacity

      method to set dust opacity model


   .. py:attribute:: free_parameters


   .. py:attribute:: Nparams_forGP
      :value: 0



   .. py:attribute:: fixed_parameters


   .. py:attribute:: r_out


   .. py:attribute:: r_out_rad


   .. py:attribute:: N_GP


   .. py:attribute:: j0_zeros


   .. py:attribute:: j0N_plus


   .. py:attribute:: j0k


   .. py:attribute:: r_GP


   .. py:attribute:: r_GP_rad


   .. py:attribute:: HT_prefactor


   .. py:attribute:: jitter
      :value: 1e-06



   .. py:attribute:: incl


   .. py:attribute:: observations


   .. py:attribute:: s_fs


   .. py:attribute:: mean_fs


   .. py:attribute:: bands
      :value: []



   .. py:attribute:: dust_params
      :value: []



   .. py:attribute:: userdef_vis_model
      :value: None



   .. py:attribute:: flux_uncert
      :value: True



   .. py:attribute:: hyperparameters_fixed
      :value: True



   .. py:method:: set_parameter(kind, free=True, dust_prop=False, GP=True, bounds=(10, 20), mean_std=(0.0, 1.0), variance=1.0, lengthscale=0.3, mean=0.0, profile=None)

      set a parameter as free or fixed.

      This method allows you to define model parameters as either free (to be inferred) or fixed (with a specified profile).

      :param kind: name of the parameter
      :type kind: str
      :param free: whether the parameter is free. Defaults to True.
      :type free: bool, optional
      :param dust_prop: whether the parameter is related to dust properties. Defaults to False.
      :type dust_prop: bool, optional
      :param GP: whether to use Gaussian process for the parameter. Defaults to True.
      :type GP: bool, optional
      :param bounds: (min, max) bounds for the parameter if GP is used. Defaults to (10, 20).
      :type bounds: tuple, optional
      :param mean_std: (mean, std) for the parameter if not using GP. Defaults to (0.0, 1.0).
      :type mean_std: tuple, optional
      :param variance: variance for the GP kernel. Defaults to 1.0.
      :type variance: float, optional
      :param lengthscale: lengthscale for the GP kernel. Defaults to 0.3.
      :type lengthscale: float, optional
      :param mean: mean for the GP prior. Defaults to 0.0.
      :type mean: float, optional
      :param profile: fixed profile for the parameter if not free. Defaults to None.
      :type profile: array or function, optional



   .. py:method:: set_observations(band, q, V, s, f_s, f_mean, nu, Nch)

      set observations for a given band.

      This method allows you to input observational data for a specific band, including spatial frequencies, visibilities, uncertainties, flux scaling factors, and frequencies.

      :param band: name of the observation band
      :type band: str
      :param q: spatial frequencies in 1/arcsec. Should be a dictionary of arrays for each channel.
      :type q: dict
      :param V: observed visibilities in Jy. Should be a dictionary of arrays for each channel.
      :type V: dict
      :param s: uncertainties in Jy. Should be a dictionary of arrays for each channel.
      :type s: dict
      :param f_s: flux scaling factor standard deviation
      :type f_s: float
      :param f_mean: mean flux scaling factor
      :type f_mean: float
      :param nu: frequencies in Hz
      :type nu: array
      :param Nch: number of channels. Shound be consistent with q, V, s.
      :type Nch: int



   .. py:method:: set_opacity(opac_dict, Na=1000, Nq=1000, smooth=True, log10_a_smooth=0.05, a_min=None, a_max=None)

      Set dust opacity model.

      This method allows you to define the dust opacity model using precomputed opacity data.

      :param opac_dict: dictionary containing dust opacity data with keys 'a', 'lam', 'k_abs', 'k_sca', 'g'
      :type opac_dict: dict
      :param Na: number of dust size grid points. Defaults to 1000.
      :type Na: int, optional
      :param Nq: number of size distribution index grid points. Defaults to 1000.
      :type Nq: int, optional
      :param smooth: whether to apply smoothing to the opacity table. Defaults to True.
      :type smooth: bool, optional
      :param log10_a_smooth: smoothing scale in log10 grain size. Defaults to 0.05.
      :type log10_a_smooth: float, optional
      :param a_min: minimum grain size in microns. Defaults to the minimum grain size in the opacity data.
      :type a_min: float, optional
      :param a_max: maximum grain size in microns. Defaults to the maximum grain size in the opacity data.
      :type a_max: float, optional



   .. py:method:: calc_model(parameters)

      Compute model visibilities using the current parameters.

      This method calculates the model visibilities and intensities for all observations based on the provided parameters.

      :param parameters: dictionary of model parameters. Should contain keys corresponding to free and fixed parameters.
      :type parameters: dict

      :returns: dictionary of model visibilities for each band and observation.
                I_res (dict): dictionary of model intensities for each band and observation.
      :rtype: V_res (dict)



.. py:class:: observation(name, nu, q, V, s)

   .. py:attribute:: name


   .. py:attribute:: nu


   .. py:attribute:: q


   .. py:attribute:: V


   .. py:attribute:: s


.. py:class:: inference(model)

   class for inference methods

   .. attribute:: prior

      method to show prior distributions

   .. attribute:: SVI_MAP

      method to run SVI for MAP estimation

   .. attribute:: MCMC

      method to run MCMC sampling


   .. py:attribute:: model


   .. py:method:: prior(num_samples=1, seed=None)

      Show prior distributions for the latent parameters.

      This method generates samples from the prior distributions of the latent parameters defined in the model.

      :param num_samples: number of prior samples to generate. Defaults to 1.
      :type num_samples: int, optional
      :param seed: random seed for reproducibility. Defaults to None.
      :type seed: int, optional

      :returns: results object containing prior samples and related information.
      :rtype: prior (results)



   .. py:method:: SVI_MAP(num_iterations=1000, num_particles=1, adam_lr=0.01, uniform_radius=0.1, seed=None)

      Run Stochastic Variational Inference (SVI) to find the Maximum A Posteriori (MAP) estimate of the latent parameters.

      This method uses SVI with an AutoDelta guide to estimate the MAP of the latent parameters defined in the model.

      :param num_iterations: number of SVI iterations. Defaults to 1000
      :type num_iterations: int, optional
      :param num_particles: number of particles for ELBO estimation. Defaults to 1.
      :type num_particles: int, optional
      :param adam_lr: learning rate for the Adam optimizer. Defaults to 0.01.
      :type adam_lr: float, optional
      :param uniform_radius: radius for uniform initialization of parameters. Defaults to 0.1.
      :type uniform_radius: float, optional
      :param seed: random seed for reproducibility. Defaults to None.
      :type seed: int, optional

      :returns: results object containing MAP estimates and related information.
      :rtype: svi_map_results (results)



   .. py:method:: MCMC(num_warmup, num_samples, step_size=1.0, num_chains=1, max_tree_depth=10, adapt_step_size=True, uniform_radius=0.1, seed=None)

      Run MCMC sampling using the NUTS algorithm.

      This method performs MCMC sampling to estimate the posterior distributions of the latent parameters defined in the model.

      :param num_warmup: number of warmup iterations.
      :type num_warmup: int
      :param num_samples: number of MCMC samples to draw.
      :type num_samples: int
      :param step_size: initial step size for the NUTS sampler. Defaults to 1.0.
      :type step_size: float, optional
      :param num_chains: number of MCMC chains to run in parallel. Defaults to 1.
      :type num_chains: int, optional
      :param max_tree_depth: maximum tree depth for the NUTS sampler. Defaults to 10.
      :type max_tree_depth: int, optional
      :param adapt_step_size: whether to adapt the step size during warmup. Defaults to True.
      :type adapt_step_size: bool, optional
      :param uniform_radius: radius for uniform initialization of parameters. Defaults to 0.1.
      :type uniform_radius: float, optional
      :param seed: random seed for reproducibility. Defaults to None.
      :type seed: int, optional

      :returns: results object containing posterior samples and related information.
      :rtype: mcmc_results (results)



.. py:class:: results(r, sample, logP)

   class for storing inference results

   .. attribute:: r

      radial grid points

   .. attribute:: sample

      dictionary of samples

   .. attribute:: logP

      dictionary of log probabilities


   .. py:attribute:: r


   .. py:attribute:: sample


   .. py:attribute:: logP


.. py:class:: plot(results)

   class for plotting results
   .. attribute:: sample_paths

      method to plot sample paths


   .. py:attribute:: r


   .. py:attribute:: sample


   .. py:method:: sample_paths(key, nskip=100, plot_kwargs={'alpha': 0.5, 'lw': 1.0, 'color': 'royalblue'}, scatter_kwargs={'alpha': 0.5, 'color': 'royalblue'})

      Plot sample paths for a given parameter key.

      :param key: parameter key to plot
      :type key: str
      :param nskip: number of samples to skip for plotting. Defaults to 100.
      :type nskip: int, optional
      :param plot_kwargs: keyword arguments for line plots. Defaults to {'alpha': 0.5, 'lw': 1.0, 'color':'roayalblue'}.
      :type plot_kwargs: dict, optional
      :param scatter_kwargs: keyword arguments for scatter plots. Defaults to {'alpha': 0.5, 'color':'royalblue'}.
      :type scatter_kwargs: dict, optional

      :returns: None



