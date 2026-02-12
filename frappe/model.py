#########################################################################
### FRAPPE: Flexible Radial Analysis of ProtoPlanetary disk Emissions ###
#########################################################################


import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import jax.random as random
import numpy as np
import warnings
import numpy as np
from scipy.special import j0


from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.signal import convolve as jax_convolve

import numpyro
from numpyro.distributions import Normal, Uniform, MultivariateNormal
from numpyro.infer import MCMC as numpyro_MCMC
from numpyro.infer import NUTS, init_to_median, init_to_uniform
import matplotlib.pyplot as plt
from numpyro.infer import Predictive
from numpyro.infer import SVI, Trace_ELBO, init_to_sample
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoDelta


from ._constants import *
from ._utilities import *

from scipy.special import j0 as J0
from scipy.special import j1 as J1
from scipy.special import jn_zeros

from jax.debug import print as jax_print

jax.config.update("jax_enable_x64", True) 




class model:
    '''class for FRAPPE model

    Attributes:
        set_parameters: method to set model parameters
        set_observations: method to set observational data
        set_opacity: method to set dust opacity model
    '''

    def __init__(self, incl, r_out, N_GP, spacing = 'FB', r_in = None, userdef_vis_model = None, flux_uncert = True, jitter=1e-4, hyperparameters_fixed=True):
        '''initialize the model

        Args:
            incl (float): inclination angle in degrees
            r_out (float): outer radius in arcseconds
            N_GP (int): number of Gaussian process points
            userdef_vis_model (function, optional): user-defined visibility model function. Defaults to None. The function should take (V, obs, f_latents) as arguments and return modified V.
            flux_uncert (bool, optional): whether to include flux uncertainty. Defaults to True.
            jitter (float, optional): jitter value for numerical stability. Defaults to 1e-6.
            hyperparameters_fixed (bool, optional): whether to fix hyperparameters. Defaults to True.

        '''
        
        self._free_parameters = {}
        self._Nparams_forGP = 0
        self._fixed_parameters = {}

        self._r_out = r_out
        self._r_out_rad = jnp.deg2rad( self._r_out/3600 )

        self._N_GP = N_GP

        if spacing == 'FB':
            self._j0_zeros = jn_zeros(0, self._N_GP + 1)
            self._j0N_plus = self._j0_zeros[-1]
            self._j0k = self._j0_zeros[:-1]
            
            self._r_GP = self._r_out * self._j0k / self._j0N_plus
            self._r_GP_rad = np.deg2rad(self._r_GP/3600)

            self._HT_prefactor = 4.0 * np.pi * self._r_out_rad ** 2/ (self._j0N_plus ** 2 * J1(self._j0k) ** 2)

        elif spacing == 'linear':
            
            warnings.warn('Please use spacing = \'FB\' if you work with the visibilities', UserWarning)

            self._r_GP = jnp.linspace( r_in, self._r_out, self._N_GP )

            self._dr = self._r_GP[1] - self._r_GP[0]
            self._r_GP_rad = jnp.deg2rad( self._r_GP/3600 )

            self._HT_prefactor = 2.0 * np.pi * (self._r_out_rad / self._N_GP) **2
   
        self._jitter = jitter

        self._incl = np.deg2rad(incl)

        
        self._observations = {}
        self._s_fs = {}
        self._mean_fs = {}
        self._bands = []


        self._userdef_vis_model = userdef_vis_model

        self._flux_uncert = flux_uncert

        self._hyperparameters_fixed = hyperparameters_fixed

                

        
    def _set_latent_params( self, calc_cond_num = False ):
        '''
        Set latent parameters using Gaussian processes.
        Returns a dictionary of latent parameters.
        '''
    
        R = self._r_GP[:, None]

        if not self._hyperparameters_fixed:

            priors = self._free_parameters['log10_Sigma_d']

            _g_variance = priors['g_variance'] + (numpyro.sample(f"variance", Uniform(-1.0, 1.0)) + 1.0) / 2.0 * ( 1.0 - priors['g_variance'] )
            _g_lengthscale = priors['g_lengthscale'] + (numpyro.sample(f"lengthscale", Uniform(-1.0, 1.0)) + 1.0) / 2.0 * ( jnp.max(R)/4.0 - priors['g_lengthscale'] )
            _g_mean = priors['g_mean']

        
        f_latents = {}
    
        
        for param_name, priors in self._free_parameters.items():

            if priors['GP'] == True:
                
                if self._hyperparameters_fixed:
                    _g_variances = priors['g_variance']
                    _g_lengthscales = priors['g_lengthscale']

                    _g_mean = priors['g_mean']

                    L_K = priors['L_K']

                else:
            
                    K = jnp.zeros((R.shape[0], R.shape[0]))

                    for  _g_variance, _g_lengthscale in zip( _g_variances, _g_lengthscales ):
                        K += rbf_kernel(R, R, _g_variance, _g_lengthscale)

                    relative_jitter = jnp.nanmean(jnp.diag(K)) * self._jitter
                    K += jnp.eye(R.shape[0]) * relative_jitter


                    if calc_cond_num:
                        eigenvalues = jnp.linalg.eigvalsh(K)
                        cond_num = jnp.max(eigenvalues) / jnp.maximum(jnp.min(eigenvalues), 1e-18)
                        jax_print("Param: {name}, Condition Number: {cond}", name=param_name, cond=cond_num)
                    
                                    
                    L_K = jnp.linalg.cholesky(K)

                
                z = numpyro.sample(
                    f'g_{param_name}_z',
                    Normal(0.0, 1.0).expand([R.shape[0]])
                )

                _g_latent = _g_mean + L_K @ z

                numpyro.deterministic(f'g_{param_name}', _g_latent)
                
                penalty = check_range(_g_latent, lower=-5.0, upper=5.0, alpha=10.0)
                numpyro.factor(f"penalty_{param_name}", jnp.sum(penalty))

                f_latents[param_name] = sigmoid_transform(
                    _g_latent,
                    min_val=priors['f_min'],
                    max_val=priors['f_max']
                )

                                
                    
            else:
                f_latents[param_name] = numpyro.sample(
                                f'g_{param_name}',
                                Normal( 0.0, 1.0 )
                            ) * priors['f_s'] + priors['f_mean']
                

                numpyro.deterministic(f'{param_name}', f_latents[param_name])

        
        for param_name, profile in self._fixed_parameters.items():
            f_latents[param_name] = profile['profile']
        
    
        return f_latents


    def set_parameter(self, kind, free = True, GP =True, 
                      bounds = (10, 20), mean_std = (0.0, 1.0),
                      variance = 1.0, lengthscale = 0.3, mean = 0.0, 
                      profile = None):
        '''set a parameter as free or fixed.

        This method allows you to define model parameters as either free (to be inferred) or fixed (with a specified profile).
        
        Args:
            kind (str): name of the parameter
            free (bool, optional): whether the parameter is free. Defaults to True.
            dust_prop (bool, optional): whether the parameter is related to dust properties. Defaults to False.
            GP (bool, optional): whether to use Gaussian process for the parameter. Defaults to True.
            bounds (tuple, optional): (min, max) bounds for the parameter if GP is used. Defaults to (10, 20).
            mean_std (tuple, optional): (mean, std) for the parameter if not using GP. Defaults to (0.0, 1.0).
            variance (float, optional): variance for the GP kernel. Defaults to 1.0.
            lengthscale (float, optional): lengthscale for the GP kernel. Defaults to 0.3.
            mean (float, optional): mean for the GP prior. Defaults to 0.0.
            profile (array or function, optional): fixed profile for the parameter if not free. Defaults to None.

        '''
    
        
        if free:

            _g_variances = np.atleast_1d(variance)
            _g_lengthscales = np.atleast_1d(lengthscale)

            R = self._r_GP[:, None]
            K = jnp.zeros((R.shape[0], R.shape[0]))

            for  _g_variance, _g_lengthscale in zip( _g_variances, _g_lengthscales ):
                K += rbf_kernel(R, R, _g_variance, _g_lengthscale)

            relative_jitter = jnp.nanmean(jnp.diag(K)) * self._jitter
            K += jnp.eye(R.shape[0]) * relative_jitter

       
            L_K = jnp.linalg.cholesky(K)

            self._free_parameters[kind] = { 'f_min' : bounds[0], 'f_max' : bounds[1], 'GP' : GP,
                                          'f_s' : mean_std[1], 'f_mean' : mean_std[0],
                                           'g_variance':np.atleast_1d(variance),
                                           'g_lengthscale':np.atleast_1d(lengthscale), 'L_K' : L_K,
                                           'g_mean':mean }
            
            if GP:
                self._Nparams_forGP += 1
            

        else:
            if profile is not None:
                
                if callable(profile):
                    self._fixed_parameters[kind] = { 'profile' : profile(self._r_GP), 'GP' : GP, }        
                else:
                    self._fixed_parameters[kind] = { 'profile' : profile, 'GP' : GP, }
            else:
                raise ValueError(f'Profile for {kind} is not set.')


    def _expansion_model( self, f_latents, obs, dryrun = False):
        '''
        Generate the expansion model for a given observation.
        f_latents: dictionary of latent parameters
        obs: observation object
        dryrun: boolean, if True, return intermediate results for debugging
        '''

        Sigma_d = 10**( f_latents['log10_Sigma_d'] )
        T = 10**( f_latents['log10_T'] )
        
        q = f_latents['q']
        log10_a_max = f_latents['log10_a_max']

        k_abs_tot = 10**obs.log10_k_abs_tot_itp( (log10_a_max, q) )
        k_sca_eff_tot = 10**obs.log10_k_sca_eff_tot_itp( (log10_a_max, q) )
        
        _I = f_I(obs._nu, self._incl, T, Sigma_d, k_abs_tot, k_sca_eff_tot )


        if obs.kind == 'visibility':

            # Hankel transform
            V = jnp.dot(obs.H, _I) / 1e-23 # Jy

            if self._userdef_vis_model is not None:
                V = self._userdef_vis_model( V, obs, f_latents )

            if dryrun:

                return V, _I

            else:

                obs.V_model = V

        elif obs.kind == 'radialprofile':

            # Convolve with beam
            I_convolved = jax_convolve( _I, obs._beam_kernel, mode='same' )
            I_convolved_itp = jnp.interp(obs._r, self._r_GP, I_convolved)

            if dryrun:

                return I_convolved_itp, _I

            else:

                obs.I_model = I_convolved_itp
            

    def _generate_model( self, f_latents ):
        '''
        Generate the model for all observations.
        f_latents: dictionary of latent parameters
        '''

        for band in self._bands:
            
            obs = self._observations[band]
            
            for _obs in obs:

                self._expansion_model( f_latents, _obs )

            

    def _GP_sample( self ):
        '''
        NumPyro model for Gaussian process sampling.
        '''

        f_latents = self._set_latent_params()
   
        self._generate_model( f_latents )

        self._sample_model( )

        


    def _sample_model( self ):
        '''
        Sample the model for all observations.
        '''
        
        for band in self._bands:
            
            obs = self._observations[band]

            if self._flux_uncert:

                
                '''    
                _g_f_band = numpyro.sample(
                            f"g_f_scale_{band}",
                            TruncatedNormal(
                                loc=0,
                                scale = 1.0,
                                low= -self._s_fs_max_sigma[band],
                                high= self._s_fs_max_sigma[band]
                            )
                        )

                '''

                '''        
                f_band = sigmoid_transform(
                    _g_f_band,
                    min_val= 1.0 - 3*self._s_fs[band], 
                    max_val= 1.0 + 3*self._s_fs[band]
                )
                '''
                # f_band = 1.0 + _g_f_band*self._s_fs[band]


                # 標準正規分布からサンプリング
                _g_f_band = numpyro.sample(
                    f"g_f_scale_{band}",
                    Normal(0, 1)
                )

                # 指数変換することで対数正規分布にする
                # f_band = exp(0 + 0.1 * standard_normal)
                f_band = jnp.exp( jnp.log(self._mean_fs[band]) + _g_f_band * self._s_fs[band])


                numpyro.deterministic(f"f_band_{band}", f_band)

                for _obs in obs:

                    if _obs.kind == 'visibility':

                        # If you want to fit the visibility dicretly....

                        V_final = _obs.V_model / f_band 

                        numpyro.deterministic(f"V_final_{_obs._name}", V_final)

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    Normal(loc= V_final, scale= _obs._s),
                                    obs = _obs._V
                                )
                    elif _obs.kind == 'radialprofile':

                        # Instead if you fit the radial profiles...

                        Tb_final = I2Tb( _obs._nu, _obs.I_model ) / f_band 

                        numpyro.deterministic(f"Tb_final_{_obs._name}", Tb_final)

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    MultivariateNormal(loc= Tb_final, scale_tril=_obs.L_Cov),
                                    obs = _obs._Tb
                                )

            else:
                
                for _obs in obs:

                    if _obs.kind == 'visibility':

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    Normal(loc= _obs.V_model, scale= _obs._s),
                                    obs = _obs._V
                                )
                        
                    elif _obs.kind == 'radialprofile':

                        numpyro.sample(
                                    f"Y_observed_{_obs._name}",
                                    MultivariateNormal(loc= I2Tb( _obs._nu, _obs.I_model), scale_tril=_obs.L_Cov),
                                    obs = _obs._Tb
                                )

    def set_visibility( self, band, q, V, s, f_s, f_mean,  nu, Nch ):
        '''set observations for a given band.

        This method allows you to input observational data for a specific band, including spatial frequencies, visibilities, uncertainties, flux scaling factors, and frequencies.
        
        Args:
            band (str): name of the observation band
            q (dict): spatial frequencies in 1/arcsec. Should be a dictionary of arrays for each channel.
            V (dict): observed visibilities in Jy. Should be a dictionary of arrays for each channel.
            s (dict): uncertainties in Jy. Should be a dictionary of arrays for each channel.
            f_s (float): flux scaling factor standard deviation
            f_mean (float): mean flux scaling factor
            nu (array): frequencies in Hz
            Nch (int): number of channels. Shound be consistent with q, V, s.

        '''

        obs_tmp = []

        for nch in range(Nch):
            
            _obs = observation( f'{band}_ch_{nch}', nu[nch], kind='visibility' )
            _obs._visibility( q[nch], V[nch], s[nch] )

            #_obs.r_rad = jnp.arange( jnp.min(jnp.deg2rad(self._r_GP/3600)), jnp.max(jnp.deg2rad(self._r_GP/3600)), 1/jnp.max(_obs.q)/self.ndr )

            #kr_matrix = _obs.q[:, jnp.newaxis] * _obs.r_rad[jnp.newaxis, :]
        

            arg = 2.0 * jnp.pi* _obs._q[:, None] * self._r_out_rad * self._j0k[None, :] / self._j0N_plus

            H = self._HT_prefactor * J0(arg)
 
            _obs.H = H
            
            obs_tmp.append( _obs )
            
        self._bands.append(band)
            
        self._observations[band] = obs_tmp
        self._s_fs[band] = f_s
        self._mean_fs[band] = f_mean

    def set_radialprofile(  self, band, r, Tb, s, f_s, f_mean, nu, Nch, FWHM): 

        obs_tmp = []

        for nch in range(Nch):
            
            _obs = observation( f'{band}_ch_{nch}', nu[nch], kind='radialprofile' )
            _obs._radialprofile( r[nch], Tb[nch], s[nch], FWHM, self._dr )
            
            obs_tmp.append( _obs )
            
        self._bands.append(band)
            
        self._observations[band] = obs_tmp
        self._s_fs[band] = f_s
        self._mean_fs[band] = f_mean


    def set_opacity( self, opac_dict, Na = 1000, Nq = 1000, smooth = True, log10_a_smooth = 0.05, a_min = None, a_max = None ):
        '''Set dust opacity model.
        
        This method allows you to define the dust opacity model using precomputed opacity data.

        Args:
            opac_dict (dict): dictionary containing dust opacity data with keys 'a', 'lam', 'k_abs', 'k_sca', 'g'
            Na (int, optional): number of dust size grid points. Defaults to 1000.
            Nq (int, optional): number of size distribution index grid points. Defaults to 1000.
            smooth (bool, optional): whether to apply smoothing to the opacity table. Defaults to True.
            log10_a_smooth (float, optional): smoothing scale in log10 grain size. Defaults to 0.05.
            a_min (float, optional): minimum grain size in microns. Defaults to the minimum grain size in the opacity data.
            a_max (float, optional): maximum grain size in microns. Defaults to the maximum grain size in the opacity data. 
        '''
        
        a      = opac_dict['a']
        lam    = opac_dict['lam']
        k_abs  = opac_dict['k_abs']
        k_sca  = opac_dict['k_sca']
        gsca   = opac_dict['g']

        one_min_g = (1 - gsca)
        one_min_g[ one_min_g < 0 ] = 1e-6

        k_sca_eff = one_min_g * k_sca

        if a_min is not None and a_max is not None:
            a_dense = jnp.array(jnp.logspace( np.log10(a_min), np.log10(a_max), Na ))
        else:
            a_dense = jnp.array(jnp.logspace( np.log10(jnp.min(a)), np.log10(jnp.max(a)), Na ))

        log10_a_dense = jnp.log10(a_dense)

        q_dense = jnp.linspace( 0.0, 6.0, Nq )


        # We tried to calc kappa_tot from kappa at a single dust size at each iteration
        # to allow the opacity model to be flexible, but it did not work well probably because of the large correlation between parameters.
        # We therefore pre-calculate interpolators for each observation here.

        

        for band in self._bands:
            
            obs = self._observations[band]
            
            for _obs in obs:

                lam0 = c/_obs._nu

                log10_k_abs_tot, log10_k_sca_eff_tot = create_opacity_table( lam, a, k_abs, k_sca_eff, lam0, log10_a_dense, q_dense, smooth=smooth, 
                                                                            log10_a_smooth=log10_a_smooth, log10_a_min = jnp.log10(a_dense[0]) )

                
                
                _obs.log10_a_dense = log10_a_dense
                _obs.q_dense = q_dense
                _obs.log10_k_abs_tot = log10_k_abs_tot
                _obs.log10_k_sca_eff_tot = log10_k_sca_eff_tot


                _obs.log10_k_abs_tot_itp = RegularGridInterpolator(
                                            (log10_a_dense, q_dense),
                                            log10_k_abs_tot,
                                            method='linear',
                                            bounds_error=False,
                                            fill_value=None
                                        )
                
                _obs.log10_k_sca_eff_tot_itp = RegularGridInterpolator(
                                            (log10_a_dense, q_dense),
                                            log10_k_sca_eff_tot,
                                            method='linear',
                                            bounds_error=False,
                                            fill_value=None
                                        )
                

    def calc_model( self, parameters ):
        """Compute model visibilities using the current parameters.

        This method calculates the model visibilities and intensities for all observations based on the provided parameters.

        Args:
            parameters (dict): dictionary of model parameters. Should contain keys corresponding to free and fixed parameters.

        Returns:
            V_res (dict): dictionary of model visibilities for each band and observation.
            I_res (dict): dictionary of model intensities for each band and observation.

        """

        f_latents = parameters


        #if plot:
        #   fig, axes = plt.subplots(  N_panels, 2 , figsize=(15, 5*N_panels) )

        V_res = {}
        I_res = {}
        
        
        for band in self._bands:
            
            V_res[band] = {}
            I_res[band] = {}
            
            obs = self._observations[band]

            for _obs in obs:

                #DP = self.dp

                #DP.debug_time['t0'].append( time.perf_counter() )

                _V_res , _I_res= self._expansion_model( f_latents, _obs, dryrun=True )



                V_res[band][_obs._name] = jnp.array(_V_res)
                I_res[band][_obs._name] = jnp.array(_I_res)


        return V_res, I_res
    

    def _g_to_f( self, g_samples ):

         # 物理パラメータ (f_samples) への変換処理
        f_samples = {}
        for param_name, priors in self._free_parameters.items():
            if priors['GP'] == False:
                f_samples[param_name] = g_samples[f'{param_name}']
            else:
                f_samples[param_name] = sigmoid_transform(
                    g_samples[f'g_{param_name}'], 
                    min_val=priors['f_min'], max_val=priors['f_max'])
        return f_samples


class observation:
    '''class for observation data

    '''

    def __init__(self, name, nu, kind='visibility' ):
        '''initialize observation data
        
        Args:
            name (str): name of the observation
            nu (float): frequency in Hz
            q (array): spatial frequencies in 1/arcsec
            V (array): observed visibilities in Jy
            s (array): uncertainties in Jy
        '''
        self.kind = kind
        self._name = name
        self._nu = nu

    def _visibility(self, q, V, s):
        '''set visibility data
        
        Args:
            q (array): spatial frequencies in 1/arcsec
            V (array): observed visibilities in Jy
            s (array): uncertainties in Jy
        '''
        self._q =  jax.device_put(jnp.asarray(q))
        self._V =  jax.device_put(jnp.asarray(V))
        self._s =  jax.device_put(jnp.asarray(s))

    def _radialprofile( self, r, Tb, s, FWHM, dr):

        self._r = jax.device_put(jnp.asarray(r))
        self._Tb = jax.device_put(jnp.asarray(Tb))
        self._s = jax.device_put(jnp.asarray(s))
        self._FWHM = FWHM


        ## Covariance matrix
        sigma_beam = FWHM / (2 * jnp.sqrt(2 * jnp.log(2)))
                    
        dist_sq = (r[:, None] - r[None, :])**2
                    
        R_dist = jnp.exp(-dist_sq / (4 * sigma_beam**2))
        

        self.Cov = s[:, None] * R_dist * s[None, :]
                    
        
        self.Cov += jnp.eye(self.Cov.shape[0]) * 1e-6

        self.L_Cov = jnp.linalg.cholesky(self.Cov)


        ## beam kernel

        
        
        kernel_r = jnp.arange(-4 * sigma_beam, 4 * sigma_beam + dr, dr)
        kernel = jnp.exp(-(kernel_r**2) / (2.0 * sigma_beam**2))
        kernel = kernel / jnp.sum(kernel)

        self._beam_kernel = kernel
    
        

class inference:
    '''class for inference methods
    
    Attributes:
        prior: method to show prior distributions
        SVI_MAP: method to run SVI for MAP estimation
        MCMC: method to run MCMC sampling
    '''

    def __init__(self, model ):

        self.model = model


    def prior(self, num_samples = 1, seed = None, calc_cond_num = False):
        '''Show prior distributions for the latent parameters.

        This method generates samples from the prior distributions of the latent parameters defined in the model.

        Args:
            num_samples (int, optional): number of prior samples to generate. Defaults to 1.
            seed (int, optional): random seed for reproducibility. Defaults to None.

        Returns:
            prior (results): results object containing prior samples and related information.

        '''

        if seed is None:
            seed = np.random.randint(0, 1e6)

        def prior_model():
            f_latents = self.model._set_latent_params(calc_cond_num=calc_cond_num)

            return f_latents


        

        prior_predictive = Predictive(prior_model, num_samples=num_samples)


        rng_key = jax.random.PRNGKey(seed)


        f_func_all = {}
        g_func_all = {}


        for i, (param_name, priors) in enumerate(self.model._free_parameters.items()):

            rng_key, rng_key2 = jax.random.split(rng_key)
            prior_predictions = prior_predictive(rng_key)[f'g_{param_name}']

            if priors['GP'] == True:

                for j, g_func in enumerate(prior_predictions):

                    f_func = sigmoid_transform(g_func, 
                                            min_val=priors['f_min'], 
                                            max_val=priors['f_max'])
                    
                    
                    if j == 0:
                        _f_func_all = f_func
                        _g_func_all = g_func
                    else:
                        _f_func_all = jnp.vstack((_f_func_all, f_func))
                        _g_func_all = jnp.vstack((_g_func_all, g_func))
                    

                f_func_all[param_name] = jnp.array([_f_func_all])
                g_func_all[param_name] = jnp.array([_g_func_all])

            else:

                f_func = prior_predictions * priors['f_s'] + priors['f_mean']

                f_func_all[param_name] = jnp.array([f_func])
                g_func_all[param_name] = jnp.array([prior_predictions])


        self._prior = results(  r = self.model._r_GP, 
                                    sample = { 'prior_f': f_func_all,
                                               'prior_g': g_func_all,
                                            },
                                    logP = { 'prior': None }
                                    )
        

        return self._prior
    
    def SVI_MAP(self, num_iterations=1000, num_particles=1, adam_lr=0.01, uniform_radius = 0.1, seed = None):
        '''Run Stochastic Variational Inference (SVI) to find the Maximum A Posteriori (MAP) estimate of the latent parameters.
        
        This method uses SVI with an AutoDelta guide to estimate the MAP of the latent parameters defined in the model.

        Args:
            num_iterations (int, optional): number of SVI iterations. Defaults to 1000
            num_particles (int, optional): number of particles for ELBO estimation. Defaults to 1.
            adam_lr (float, optional): learning rate for the Adam optimizer. Defaults to 0.01.
            uniform_radius (float, optional): radius for uniform initialization of parameters. Defaults to 0.1.
            seed (int, optional): random seed for reproducibility. Defaults to None.

        Returns:
            svi_map_results (results): results object containing MAP estimates and related information.

        '''

        if seed is None:
            seed = np.random.randint(0, 1e6)
        
        rng_key = jax.random.PRNGKey(seed)
        
        _init_strategy = init_to_uniform(radius=uniform_radius)

         
        guide = AutoDelta(self.model._GP_sample, init_loc_fn=_init_strategy)
        optimizer = Adam(adam_lr)
        elbo = Trace_ELBO(num_particles)
        svi = SVI(self.model._GP_sample, guide, optimizer, elbo)
        
        # run svi
        rng_key, rng_key_svi = jax.random.split(rng_key)
        svi_result = svi.run(
                rng_key_svi, 
                num_iterations, 
                progress_bar=True
            )
        
        self.svi_result = svi_result
        params = svi_result.params
        loss = svi_result.losses[-1]


        # 1. Remap internal AutoDelta names to the original sampling names in the model
        map_samples_fixed = {}
        for k, v in params.items():
            name_in_model = k.replace("_auto_loc", "")
            map_samples_fixed[name_in_model] = v[None, ...]

        # 2. Recalc the model outputs at the MAP estimate
        predictive = Predictive(self.model._GP_sample, posterior_samples=map_samples_fixed, num_samples=1)
        rng_key, rng_key_pred = jax.random.split(rng_key)
        map_outputs = predictive(rng_key_pred)
        
        
        map_estimates = {k: jnp.array([[jnp.squeeze(v, axis=0)]]) for k, v in map_outputs.items()}
        

        self.delta_medians = {}
        for param_name, priors in self.model._free_parameters.items():
            if priors['GP'] == False:
                self.delta_medians[param_name] = map_estimates[param_name]
            else:
                # 潜在変数 g_... を取得。
                # predictiveを通しているので map_estimates[f'g_{param_name}'] が存在する
                g_predictions = map_estimates[f'g_{param_name}']
                
                # 物理領域への変換
                f_predictions = sigmoid_transform(
                    g_predictions, 
                    min_val=priors['f_min'], 
                    max_val=priors['f_max']
                )
                self.delta_medians[param_name] = f_predictions

        self.svi_map_results = results(  r = self.model._r_GP, 
                                    sample = { 'MAP_g': map_estimates,
                                               'MAP_f': self.delta_medians,
                                            },
                                    logP = { 'MAP': -loss }
                                    )

        return self.svi_map_results


    def MCMC(self, num_warmup, num_samples, step_size = 1.0, num_chains = 1, max_tree_depth=10, adapt_step_size=True, uniform_radius = 0.1, seed = None):
        '''Run MCMC sampling using the NUTS algorithm.
        
        This method performs MCMC sampling to estimate the posterior distributions of the latent parameters defined in the model.

        Args:
            num_warmup (int): number of warmup iterations.
            num_samples (int): number of MCMC samples to draw.
            step_size (float, optional): initial step size for the NUTS sampler. Defaults to 1.0.
            num_chains (int, optional): number of MCMC chains to run in parallel. Defaults to 1.
            max_tree_depth (int, optional): maximum tree depth for the NUTS sampler. Defaults to 10.
            adapt_step_size (bool, optional): whether to adapt the step size during warmup. Defaults to True.
            uniform_radius (float, optional): radius for uniform initialization of parameters. Defaults to 0.1.
            seed (int, optional): random seed for reproducibility. Defaults to None.

        Returns:
            mcmc_results (results): results object containing posterior samples and related information.
        '''

        if seed is None:
            seed = np.random.randint(0, 1e6)
        
        rng_key = jax.random.PRNGKey(seed)

        '''
        if init_strategy == 'value':
            _init_strategy = init_to_value( values=medians )
            init_params = medians
        elif init_strategy == 'uniform':
            _init_strategy = init_to_uniform(radius=radius)
            init_params = None
        elif init_strategy == 'sample':
            _init_strategy = init_to_sample()
            init_params = None
        elif init_strategy == 'median':
            _init_strategy = init_to_median()
            init_params = None
        '''

        _init_strategy = init_to_uniform(radius=uniform_radius)

        kernel = NUTS(self.model._GP_sample,
                        step_size=step_size,
                        adapt_step_size=adapt_step_size,
                        dense_mass=True,
                        init_strategy = _init_strategy,
                        max_tree_depth=max_tree_depth)
            
        mcmc = numpyro_MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
            chain_method='parallel'
        )

        # --- warmup ---
        mcmc.warmup(
            rng_key,
            extra_fields=('diverging', 'accept_prob', 'energy', 'potential_energy'),
            collect_warmup=True
        )
        
        g_warmup_samples = mcmc.get_samples(group_by_chain=True)
        warmup_logP = -mcmc.get_extra_fields(group_by_chain=True)['potential_energy']

        # --- main sampling ---
        mcmc.run(
            mcmc.post_warmup_state.rng_key,
            init_params=None,
            extra_fields=('diverging', 'accept_prob', 'energy', 'potential_energy')
        )
        
        mcmc.print_summary()

        g_posterior_samples = mcmc.get_samples(group_by_chain=True)
        posterior_logP = -mcmc.get_extra_fields(group_by_chain=True)['potential_energy']


        f_warmup_samples = self.model._g_to_f( g_warmup_samples )
        f_posterior_samples = self.model._g_to_f( g_posterior_samples )
        

        self.mcmc_results = results(  r = self.model._r_GP, 
                                    sample = { 'warmup_f': f_warmup_samples,
                                                  'posterior_f': f_posterior_samples,
                                                   'warmup_g': g_warmup_samples,
                                                   'posterior_g': g_posterior_samples,
                                                 },
                                    logP = { 'warmup': warmup_logP,
                                             'posterior': posterior_logP }
                                    )
        
        return self.mcmc_results



class results:
    '''class for storing inference results

    Attributes:
        r: radial grid points
        sample: dictionary of samples
        logP: dictionary of log probabilities
    '''

    def __init__(self, r, sample, logP):
        self.r = r
        self.sample = sample
        self.logP = logP


class plot:
    '''class for plotting results
    Attributes:
        sample_paths: method to plot sample paths
    ''' 

    def __init__(self, results ):
        self.r =  results.r
        self.sample =  results.sample

        print('saved keys: ', self.sample.keys() )

    def _plot_axes(  self, ax, x, y, nskip, twod = True, xlabel=None, ylabel=None, plot_kwargs = {}, scatter_kwargs = {}):

        for j in range(0, y.shape[0], nskip):


            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if twod:

                ax.plot( x, y[j,:], **plot_kwargs )

            else:


                ax.scatter( np.median(x), y[j], **scatter_kwargs )




    def sample_paths( self, key, nskip = 100, plot_kwargs = {'alpha': 0.5, 'lw': 1.0, 'color':'royalblue'}, scatter_kwargs = {'alpha': 0.5, 'color':'royalblue'} ):
        '''Plot sample paths for a given parameter key.

        Args:
            key (str): parameter key to plot
            nskip (int, optional): number of samples to skip for plotting. Defaults to 100.
            plot_kwargs (dict, optional): keyword arguments for line plots. Defaults to {'alpha': 0.5, 'lw': 1.0, 'color':'roayalblue'}.
            scatter_kwargs (dict, optional): keyword arguments for scatter plots. Defaults to {'alpha': 0.5, 'color':'royalblue'}.  
            
        Returns:
            None
        ''' 

        _samples = self.sample[key]

        N_panel = len( _samples.keys() )

        fig, axes = plt.subplots(N_panel,1, figsize=(10, 4*N_panel), sharex=True )

        axes = np.atleast_1d(axes)

        for i, param_name in enumerate( _samples.keys() ):

            _sample = _samples[param_name]

            if len(np.shape(_sample)) == 3:
                _sample = _sample.reshape( (_sample.shape[0]*_sample.shape[1], _sample.shape[2]) )

                
                self._plot_axes( axes[i], self.r, _sample, nskip, twod = True, xlabel='Radius (arcsec)', ylabel=f'{param_name}', plot_kwargs = plot_kwargs, scatter_kwargs = scatter_kwargs )

            else:
                _sample = _sample[0]

                self._plot_axes( axes[i], self.r, _sample, nskip, twod = False, xlabel='', ylabel=f'{param_name}', plot_kwargs = plot_kwargs, scatter_kwargs = scatter_kwargs )



    