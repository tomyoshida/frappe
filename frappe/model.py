#########################################################################
### FRAPPE: Flexible Radial Analysis of ProtoPlanetary disk Emissions ###
#########################################################################


import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as cst


c = cst.c.cgs.value
k_B = cst.k_B.cgs.value
h = cst.h.cgs.value
au = cst.au.cgs.value
G = cst.G.cgs.value
M_sun = cst.M_sun.cgs.value
m_p = cst.m_p.cgs.value

import jax
import jax.numpy as jnp

import jax.random as random
import numpy as np
import warnings
import numpy as np
from scipy.special import j0


from jax.scipy.interpolate import RegularGridInterpolator


import numpyro
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC as numpyro_MCMC
from numpyro.infer import NUTS, init_to_median, init_to_uniform
import matplotlib.pyplot as plt
from numpyro.infer import Predictive
from numpyro.infer import SVI, Trace_ELBO, init_to_sample
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoDelta


from .constants import *
from .utilities import *

from scipy.special import j0 as J0
from scipy.special import j1 as J1
from scipy.special import jn_zeros


jax.config.update("jax_enable_x64", True) 




class model:

    def __init__(self, incl, r_out, N_GP, userdef_vis_model = None, flux_uncert = True, jitter=1e-6, hyperparameters_fixed=False):
        '''
        incl: inclination angle in degrees
        r_in: inner radius in arcseconds
        r_out: outer radius in arcseconds
        N_GP: number of Gaussian process points
        userdef_vis_model: function, user-defined numpyro model to modify the visibility in Jy. Input: (V (Jy), nu (Hz)). For instance, to add free-free emission.
        flux_uncert: boolean, whether to include flux calibration uncertainty
        jitter: float, jitter value for numerical stability in Gaussian processes
        '''
        
        self.free_parameters = {}
        self.Nparams_forGP = 0
        self.fixed_parameters = {}

        self.r_out = r_out
        self.r_out_rad = jnp.deg2rad( self.r_out/3600 )

        self.N_GP = N_GP
        self.j0_zeros = jn_zeros(0, self.N_GP + 1)
        self.j0N_plus = self.j0_zeros[-1]
        self.j0k = self.j0_zeros[:-1]
        
        self.r_GP = self.r_out * self.j0k / self.j0N_plus
        self.r_GP_rad = np.deg2rad(self.r_GP/3600)

        self.HT_prefactor = 4.0 * np.pi * self.r_out_rad ** 2/ (self.j0N_plus ** 2 * J1(self.j0k) ** 2)
   
        self.jitter = jitter

        self.incl = np.deg2rad(incl)

        
        self.observations = {}
        self.s_fs = {}
        self.mean_fs = {}
        self.bands = []

        self.dust_params = []

        self.userdef_vis_model = userdef_vis_model

        self.flux_uncert = flux_uncert

        self.hyperparameters_fixed = hyperparameters_fixed

                

        
    def _set_latent_params( self ):
        '''
        Set latent parameters using Gaussian processes.
        Returns a dictionary of latent parameters.
        '''
    
        R = self.r_GP[:, None]

        if not self.hyperparameters_fixed:

            priors = self.free_parameters['log10_Sigma_d']

            _g_variance = priors['g_variance'] + (numpyro.sample(f"variance", Uniform(-1.0, 1.0)) + 1.0) / 2.0 * ( 1.0 - priors['g_variance'] )
            _g_lengthscale = priors['g_lengthscale'] + (numpyro.sample(f"lengthscale", Uniform(-1.0, 1.0)) + 1.0) / 2.0 * ( jnp.max(R)/2.0 - priors['g_lengthscale'] )
            _g_mean = priors['g_mean']

        
        f_latents = {}
    
        
        for param_name, priors in self.free_parameters.items():

            if priors['GP'] == True:
                
                if self.hyperparameters_fixed:
                    _g_variance = priors['g_variance']
                    _g_lengthscale = priors['g_lengthscale']
                    _g_mean = priors['g_mean']

                
                K = rbf_kernel(R, R, _g_variance, _g_lengthscale)
                K += jnp.eye(R.shape[0]) * self.jitter
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

        
        for param_name, profile in self.fixed_parameters.items():
            f_latents[param_name] = profile['profile']
        
    
        return f_latents


    def set_parameter(self, kind, free = True,  dust_prop = False, GP =True, 
                      bounds = (10, 20), mean_std = (0.0, 1.0),
                      variance = 2.0, lengthscale = 0.3, mean = 0.0, 
                      profile = None):
        '''
        Set a parameter as free or fixed.
        kind: name of the parameter
        free: boolean, whether the parameter is free or fixed
        bounds: tuple, (min, max) bounds for the free parameter
        mean_std: tuple, (mean, std) for the free parameter
        g_variance_prior: float, prior variance for the Gaussian process
        g_lengthscale_prior: float, prior lengthscale for the Gaussian process
        profile: function or array, fixed profile for the parameter if not free
        '''
    
        
        if free:

            self.free_parameters[kind] = { 'f_min' : bounds[0], 'f_max' : bounds[1], 'GP' : GP,
                                          'f_s' : mean_std[1], 'f_mean' : mean_std[0],
                                           'g_variance':variance,
                                           'g_lengthscale':lengthscale, 
                                           'g_mean':mean }
            
            if GP:
                self.Nparams_forGP += 1
            

        else:
            if profile is not None:
                
                if callable(profile):
                    self.fixed_parameters[kind] = { 'profile' : profile(self.r_GP), 'GP' : GP, }        
                else:
                    self.fixed_parameters[kind] = { 'profile' : profile, 'GP' : GP, }
            else:
                raise ValueError(f'Profile for {kind} is not set.')

        if dust_prop:
            self.dust_params.append( kind )

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
        
        _I = f_I(obs.nu, self.incl, T, Sigma_d, k_abs_tot, k_sca_eff_tot )

        # Hankel transform
        V = jnp.dot(obs.H, _I) / 1e-23 # Jy

        if self.userdef_vis_model is not None:
            V = self.userdef_vis_model( V, obs, f_latents )

        if dryrun:

            return V, _I

        else:

            obs.V_model = V
            

    def _generate_model( self, f_latents ):
        '''
        Generate the model for all observations.
        f_latents: dictionary of latent parameters
        '''

        for band in self.bands:
            
            obs = self.observations[band]
            
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
        
        for band in self.bands:
            
            obs = self.observations[band]

            if self.flux_uncert:

                
                '''    
                _g_f_band = numpyro.sample(
                            f"g_f_scale_{band}",
                            TruncatedNormal(
                                loc=0,
                                scale = 1.0,
                                low= -self.s_fs_max_sigma[band],
                                high= self.s_fs_max_sigma[band]
                            )
                        )

                '''

                '''        
                f_band = sigmoid_transform(
                    _g_f_band,
                    min_val= 1.0 - 3*self.s_fs[band], 
                    max_val= 1.0 + 3*self.s_fs[band]
                )
                '''
                # f_band = 1.0 + _g_f_band*self.s_fs[band]


                # 標準正規分布からサンプリング
                _g_f_band = numpyro.sample(
                    f"g_f_scale_{band}",
                    Normal(0, 1)
                )

                # 指数変換することで対数正規分布にする
                # f_band = exp(0 + 0.1 * standard_normal)
                f_band = jnp.exp( jnp.log(self.mean_fs[band]) + _g_f_band * self.s_fs[band])


                for _obs in obs:

                    numpyro.sample(
                                f"Y_observed_{_obs.name}",
                                Normal(loc= _obs.V_model / f_band, scale= _obs.s ),
                                obs = _obs.V
                            )

            else:
                
                for _obs in obs:

                    numpyro.sample(
                                f"Y_observed_{_obs.name}",
                                Normal(loc= _obs.V_model, scale= _obs.s ),
                                obs = _obs.V
                            )

    def set_observations( self, band, q, V, s, s_f, mean_fs,  nu, Nch ):
        '''
        Set observations for a given band.
        band: string, name of the band
        q: array, spatial frequencies in 1/arcseconds
        V: array, observed visibilities in Jy
        s: array, uncertainties in visibilities in Jy
        s_f: float, fractional uncertainty in flux calibration
        mean_fs: float, mean flux scaling factor
        nu: array, observing frequencies in Hz
        Nch: int, number of channels 
        '''

        obs_tmp = []

        for nch in range(Nch):
            
            _obs = observation( f'{band}_ch_{nch}', nu[nch], q[nch], V[nch], s[nch] )

            #_obs.r_rad = jnp.arange( jnp.min(jnp.deg2rad(self.r_GP/3600)), jnp.max(jnp.deg2rad(self.r_GP/3600)), 1/jnp.max(_obs.q)/self.ndr )

            #kr_matrix = _obs.q[:, jnp.newaxis] * _obs.r_rad[jnp.newaxis, :]
        

            arg = 2.0 * jnp.pi* _obs.q[:, None] * self.r_out_rad * self.j0k[None, :] / self.j0N_plus

            H = self.HT_prefactor * J0(arg)
 
            _obs.H = H
            
            obs_tmp.append( _obs )
            
        self.bands.append(band)
            
        self.observations[band] = obs_tmp
        self.s_fs[band] = s_f
        self.mean_fs[band] = mean_fs


    def set_opacity( self, opac_dict, Na = 1000, Nq = 1000, smooth = True, log10_a_smooth = 0.05, a_min = None, a_max = None ):
        '''
        Set dust opacity model.
        opac_dict: dictionary containing dust opacity data with keys 'a', 'lam', 'k_abs', 'k_sca', 'g'
        Na: int, number of points for grain size interpolation
        Nq: int, number of points for q interpolation
        smooth: boolean, whether to apply Gaussian smoothing to the opacity data
        log10_a_smooth: float, standard deviation for Gaussian smoothing in log10 grain size 
        a_min: float, minimum grain size in microns in the opacity interpolator
        a_max: float, maximum grain size in microns in an opacity interpolator
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

        

        for band in self.bands:
            
            obs = self.observations[band]
            
            for _obs in obs:

                lam0 = c/_obs.nu

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
        '''
        Calculate model visibilities and intensities for given parameters.
        parameters: dictionary of model parameters
        Returns:
            V_res: dictionary of model visibilities for each band and observation
            I_res: dictionary of model intensities for each band and observation
        '''

        f_latents = parameters


        #if plot:
        #   fig, axes = plt.subplots(  N_panels, 2 , figsize=(15, 5*N_panels) )

        V_res = {}
        I_res = {}
        
        
        for band in self.bands:
            
            V_res[band] = {}
            I_res[band] = {}
            
            obs = self.observations[band]

            for _obs in obs:

                #DP = self.dp

                #DP.debug_time['t0'].append( time.perf_counter() )

                _V_res , _I_res= self._expansion_model( f_latents, _obs, dryrun=True )



                V_res[band][_obs.name] = jnp.array(_V_res)
                I_res[band][_obs.name] = jnp.array(_I_res)


        return V_res, I_res
    

    def g_to_f( self, g_samples ):

         # 物理パラメータ (f_samples) への変換処理
        f_samples = {}
        for param_name, priors in self.free_parameters.items():
            if priors['GP'] == False:
                f_samples[param_name] = g_samples[f'{param_name}']
            else:
                f_samples[param_name] = sigmoid_transform(
                    g_samples[f'g_{param_name}'], 
                    min_val=priors['f_min'], max_val=priors['f_max'])
        return f_samples


class observation:

    def __init__(self, name, nu, q, V, s ):

        self.name = name
        self.nu = nu
        self.q =  jax.device_put(jnp.asarray(q))
        self.V =  jax.device_put(jnp.asarray(V))
        self.s =  jax.device_put(jnp.asarray(s))
        

class inference:

    def __init__(self, model ):

        self.model = model


    def prior(self, num_samples = 1, seed = None):
        '''
        Show prior distributions for the latent parameters.
        num_samples: number of prior samples to generate
        jitter: jitter value for numerical stability
        log: boolean, whether to plot in log scale
        lw: line width for the plots
        alpha: transparency for the plots
        '''

        if seed is None:
            seed = np.random.randint(0, 1e6)

        def prior_model():
            f_latents = self.model._set_latent_params()

            return f_latents


        

        prior_predictive = Predictive(prior_model, num_samples=num_samples)


        rng_key = jax.random.PRNGKey(seed)


        f_func_all = {}
        g_func_all = {}


        for i, (param_name, priors) in enumerate(self.model.free_parameters.items()):

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


        self._prior = results(  r = self.model.r_GP, 
                                    sample = { 'prior_f': f_func_all,
                                               'prior_g': g_func_all,
                                            },
                                    logP = { 'prior': None }
                                    )
        

        return self._prior
    
    def SVI_MAP(self, num_iterations=1000, num_particles=1, adam_lr=0.01, uniform_radius = 0.1, seed = None):
        '''
        Run Stochastic Variational Inference (SVI) to find the Maximum A Posteriori (MAP) estimate of the latent parameters.
        rng_key: JAX random key
        num_iterations: number of SVI iterations
        num_particles: number of particles for ELBO estimation
        adam_lr: learning rate for the Adam optimizer
        uniform_radius: radius for uniform initialization
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
        for param_name, priors in self.model.free_parameters.items():
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

        self.svi_map_results = results(  r = self.model.r_GP, 
                                    sample = { 'MAP_g': map_estimates,
                                               'MAP_f': self.delta_medians,
                                            },
                                    logP = { 'MAP': -loss }
                                    )

        return self.svi_map_results


    def MCMC(self, num_warmup, num_samples, step_size = 0.1, num_chains = 1, max_tree_depth=10, adapt_step_size=True, uniform_radius = 0.1, seed = None):

        '''
        Run MCMC sampling using the NUTS algorithm.
        num_warmup: number of warmup steps
        num_samples: number of MCMC samples
        step_size: initial step size for NUTS
        num_chains: number of MCMC chains
        max_tree_depth: maximum tree depth for NUTS
        adapt_step_size: boolean, whether to adapt the step size during warmup
        uniform_radius: radius for uniform initialization
        seed: random seed
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


        f_warmup_samples = self.model.g_to_f( g_warmup_samples )
        f_posterior_samples = self.model.g_to_f( g_posterior_samples )
        

        self.mcmc_results = results(  r = self.model.r_GP, 
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

    def __init__(self, r, sample, logP):
        self.r = r
        self.sample = sample
        self.logP = logP


class plot:

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
        '''
        Plot sample paths from the prior or posterior samples.
        key: string, key in the sample dictionary to plot
        nskip: int, number of samples to skip for plotting
        kwargs: additional keyword arguments for matplotlib plot function
        ''' 

        _samples = self.sample[key]

        N_panel = len( _samples.keys() )

        fig, axes = plt.subplots(N_panel,1, figsize=(10, 4*N_panel), sharex=True )

        for i, param_name in enumerate( _samples.keys() ):

            _sample = _samples[param_name]

            if len(np.shape(_sample)) == 3:
                _sample = _sample.reshape( (_sample.shape[0]*_sample.shape[1], _sample.shape[2]) )

                
                self._plot_axes( axes[i], self.r, _sample, nskip, twod = True, xlabel='Radius (arcsec)', ylabel=f'{param_name}', plot_kwargs = plot_kwargs, scatter_kwargs = scatter_kwargs )

            else:
                _sample = _sample[0]

                self._plot_axes( axes[i], self.r, _sample, nskip, twod = False, xlabel='', ylabel=f'{param_name}', plot_kwargs = plot_kwargs, scatter_kwargs = scatter_kwargs )



    