
import numpy as np
import pickle
import os

from scipy.optimize import curve_fit
from ._constants import *

from tqdm import tqdm

class ms:

    '''Class for handling measurement set data using CASA tools.
    
    '''

    def __init__(self, casatools, casatasks):
        '''Class for handling measurement set data using CASA tools.
        
        Args:
            casatools: CASA casatools module.
            casatasks: CASA casatasks module.
        '''
        self.ctool = casatools
        self.ct = casatasks

    def _leastsq(self, nu, nu0, I, sigma, maxfev = 10000, rmse=True):
        
        def lin_model(x, a, b):
            return a * x + b

        x = (nu - nu0)/nu0

        normfac = np.mean(np.abs(I))
        y = I / normfac
        s = sigma / normfac

        popt, pcov = curve_fit(lin_model, x, y, p0=[0.0, 1.0], sigma= s , absolute_sigma=True, maxfev=maxfev)
            
        a, b = popt * normfac
        perr = np.sqrt(np.diag(pcov)) # 標準誤差
        std_a, std_b = perr * normfac


        if rmse:
            y_fit = lin_model(x, *popt)
            residuals = y - y_fit
            dof = len(y) - len(popt)
            rmse = np.sqrt(np.sum(residuals**2) / dof) if dof > 0 else 0

            std_b = rmse / np.sqrt(len(y)) * normfac

    
        return a, std_a, b, std_b
    
    def _leastsq_2d(self, dq, nu, nu0, I, sigma, maxfev=10000):
        """
        2D least squares fitting to the visibility data as a function of deprojected uv-distance and frequency.
        """
        
        def plane_model(coords, c0, c1, c2):
            dq, nu_ratio = coords
            return c0 * dq + c1 * nu_ratio + c2

        # Prepare independent variables
        nu_ratio = (nu - nu0) / nu0
        xdata = np.vstack((dq, nu_ratio)) # (2, N) shape

        # Scaling (for numerical stability)
        normfac = np.mean(np.abs(I))
        y = I / normfac
        s = sigma / normfac

        # Fitting
        # p0: [q coefficient, frequency gradient, intercept]
        popt, pcov = curve_fit(plane_model, xdata, y, p0=[0.0, 0.0, 1.0], 
                               sigma=s, absolute_sigma=True, maxfev=maxfev)

        # Rescale to original scale
        _, _, I0 = popt * normfac
        _, _, std_I0 = np.sqrt(np.diag(pcov)) * normfac

        return I0, std_I0
    

    def _get_visnames(self, vis):
        '''Get measurement set names from the given path.

        Args:
            vis (str): Path to the measurement set or directory containing measurement sets.
        Returns:
            tuple: A tuple containing the base name of the measurement set and an array of spectral window IDs.
        '''

        tb = self.ctool.table()
        tb.open(vis + '/DATA_DESCRIPTION')
        spw_id = tb.getcol('SPECTRAL_WINDOW_ID')
        tb.close()

        # print(spw_id)
            
        visname = os.path.splitext(os.path.basename(vis))[0]

        return visname, spw_id



    def split(self, vis, datacolumn = 'CORRECTED', dryrun=True):
        '''split the measurement set by spectral windows.

        Args:
            vis (str): Path to the measurement set.
            dryrun (bool, optional): If True, only simulate the split without executing it. Defaults to True.
        
        Returns:
            list: List of paths to the split measurement sets.
        
        '''
            
        visname, spw_id = self._get_visnames( vis )
            
        outputvis_arr = []
            
        for spw in spw_id:

            outputvis = f'./working_{visname}/spw_id_{spw}.ms'
            outputvis_arr.append(outputvis)

            if not dryrun:
                
                # check if working directory exists
                if not os.path.exists(f'./working_{visname}'):
                    os.makedirs(f'./working_{visname}')
                    print('Created working directory: ' + f'./working_{visname}')
                else:
                    # if exists, arise error
                    raise FileExistsError(f'Working directory ./working_{visname} already exists. Please remove it before running split.')

                for spw in spw_id:

                    os.system('rm -rf ' + outputvis)

                    
                    self.ct.split(
                            vis = vis,
                            outputvis = outputvis,
                            spw = f'{spw}',
                            keepflags = False,
                            datacolumn = datacolumn 
                        )
            
        return outputvis_arr
        

    def get_visibilities_singlechan(self, vis, pa, incl, FoV, nu0, maxfev = 10000, rmse=True, fit_2d=True ):
        '''get one-dimensional deprojected visibilities and uncertainty as a function of uv-distance for a single measurement sets list.

        This function processes multiple measurement sets, deprojects the visibilities based on the provided position angle and inclination, azimuthally averages the data according to the specified field of view, and fits the visibilities by a linear function to extract intensity values and their uncertainties at a reference frequency.

        Args:
            vis (list): List of paths to the measurement sets.
            pa (float): Position angle in degrees for deprojection.
            incl (float): Inclination angle in degrees for deprojection.
            FoV (float): Field of view in arcseconds to determine the bin size for azimuthal averaging.
            nu0 (float): Reference frequency in Hz for fitting the visibilities.

        Returns:
            tuple: A tuple containing:
                - q (np.ndarray): Deprojected uv-distances in 1/radians.
                - V (np.ndarray): Fitted intensity values at the reference frequency.
                - s (np.ndarray): Uncertainties associated with the fitted intensity values.
        '''

        q_all, Re_all, sigma2_all, nu_all = self._multiple_visibilities(
            vis, pa, incl
        )

        Re_all = Re_all / np.cos(np.deg2rad(incl))
        sigma2_all = sigma2_all / np.cos(np.deg2rad(incl))
        
        Re_dict, s_dict, nu_dict, q_dict, q_cent, q_mean, Nq = self._binning_q_only(
            q_all, Re_all, sigma2_all, nu_all, FoV
        )
        
        if fit_2d:
            q, V, s = self._process_q_nu_fit_2d(
                Re_dict, s_dict, nu_dict, q_dict, q_cent, Nq, nu0, maxfev, rmse
            )
        else:
            q, V, s = self._process_q_nu_fit(
                Re_dict, s_dict, nu_dict, q_mean, Nq, nu0, maxfev, rmse
            )

        return q, V, s
    

    def get_visibilities(self, vis, nu, pa, incl, FoV, output, save = True, maxfev = 10000, rmse=True, fit_2d=True):
        '''get one-dimensional deprojected visibilities and uncertainty as a function of uv-distance for multiple measurement set lists
        
        This function processes multiple sets of measurement sets, deprojects the visibilities based on the provided position angle and inclination, azimuthally averages the data according to the specified field of view, and fits the visibilities by a linear function to extract intensity values and their uncertainties at given reference frequencies.

        Args:
            vis (list): List of lists, where each sublist contains paths to measurement sets for a specific frequency channel.
            nu (list): List of reference frequencies in Hz corresponding to each set of measurement sets.
            pa (float): Position angle in degrees for deprojection.
            incl (float): Inclination angle in degrees for deprojection.
            FoV (float): Field of view in arcseconds to determine the bin size for azimuthal averaging.
            output (str): Base path for saving the output pickle file.
            save (bool, optional): If True, saves the output data to a pickle file. Defaults to True.
            fit_2d (bool, optional): If True, fits the visibilities using a 2D fitting method. Defaults to True.
        
        '''

        q = {}
        V = {}
        s = {}

        Nnu = len(nu)

        if Nnu != np.shape(vis)[0]:
            raise ValueError('Number of frequency channels does not match the number of visibility set lists.')

        for inu in range(Nnu):

            vis_list = vis[inu]
            nu0 = nu[inu]

            q[inu], V[inu], s[inu] = self.get_visibilities_singlechan(vis_list, pa, incl, FoV, nu0, maxfev, rmse, fit_2d)

        data = { 'q':q, 'V':V, 's':s, 'nu':nu, 'Nch':Nnu }

        if save:
            with open(output + '.pkl', 'wb') as f:
                pickle.dump(data, f)
            print('Saved visibility data to ' + output + '.pkl')


    def _load_visibility( self, vis ):
        
        tb = self.ctool.table()
        tb.open(vis)
        
        u,v,w = tb.getcol('UVW')
        weight = tb.getcol('WEIGHT')
        
        data = tb.getcol('DATA')
        
        _flag = tb.getcol('FLAG')
        
        tb.close()
        
        XX_flag = _flag[0]
        YY_flag = _flag[1]

        flag = XX_flag | YY_flag
        
        
        chan_freq = self._load_chan_freqs( vis )
        
        Re = 0.5*(data[0] + data[1]).real
        
        _sigma2_tmp = 1/weight
        
        _sigma2 = 0.5*( _sigma2_tmp[0] + _sigma2_tmp[1] )
        sigma2 = np.broadcast_to(_sigma2, (Re.shape[0], _sigma2.shape[0]))
        
        return u, v, Re, sigma2, chan_freq, flag

        
    def _load_chan_freqs(self, vis ):
        
        tb = self.ctool.table()
        
        tb.open(vis + '/SPECTRAL_WINDOW')
        chan_freq = tb.getcol('CHAN_FREQ')
        tb.close()
        
        return chan_freq


    def _deprojected_visibility(self, vis, pa, incl):

        u, v, Re, sigma2, chan_freq, flag = self._load_visibility( vis )

        pa = np.deg2rad(pa)
        incl = np.deg2rad(incl)

        u_rot = u * np.cos(pa) - v * np.sin(pa)
        v_rot = u * np.sin(pa) + v * np.cos(pa)

        v_deproj = v_rot 
        u_deproj = u_rot * np.cos(incl)

        q = np.sqrt( v_deproj**2 + u_deproj**2 )*1e2
        
        lam = c/chan_freq
        
        q_lam = q/lam
        
        
        return q_lam, Re, sigma2, chan_freq, flag



    def _multiple_visibilities( self, vis_arr, pa, incl ):
        
        
        q_all = np.array([])
        Re_all = np.array([])
        sigma2_all = np.array([])
        freq_all = np.array([])
        
        for vis in vis_arr:
            
            q_lam, Re, sigma2, chan_freq, flag = self._deprojected_visibility( vis, pa, incl )
            
            
            Re_m = np.ma.array(Re, mask=flag)
            q_lam_m = np.ma.array(q_lam, mask=flag)
            sigma2_m = np.ma.array(sigma2, mask=flag)


            q_flat = q_lam_m.compressed()
            Re_flat = Re_m.compressed()
            sigma2_flat = sigma2_m.compressed()

            freq_flat = np.broadcast_to(
                chan_freq, Re.shape
            )[~flag]


            q_all = np.append( q_all, q_flat )
            Re_all = np.append( Re_all, Re_flat )
            freq_all = np.append( freq_all, freq_flat )
            sigma2_all = np.append( sigma2_all, sigma2_flat )
            
        
        return q_all, Re_all, sigma2_all, freq_all


    def _binning_q_only(self, q_all, Re_all, sigma2_all, freq_all, FoV):

        dq = 1 / np.deg2rad(FoV / 3600)

        q_min = q_all.min()
        q_max = q_all.max()

        q_bins = np.arange(q_min, q_max + dq, dq)
        q_cent = 0.5 * (q_bins[:-1] + q_bins[1:])

        # 各データ点が属する q ビン番号
        q_index = np.digitize(q_all, q_bins) - 1

        Re_dict = {}
        s_dict = {}
        nu_dict = {}
        q_dict = {}
        q_mean = {}

        iq_valid = 0

        for iq in range(len(q_cent)):

            mask = q_index == iq

            if mask.sum() > 2:
                Re_dict[iq_valid] = Re_all[mask]
                s_dict[iq_valid] = np.sqrt(sigma2_all[mask])
                nu_dict[iq_valid] = freq_all[mask]
                
                #q_dict[iq_valid] = q_cent[iq]

                
                q_dict[iq_valid] = q_all[mask]

                _weights = 1.0 / sigma2_all[mask]
                q_mean[iq_valid] = np.average(q_all[mask], weights=_weights)

                iq_valid += 1

        return Re_dict, s_dict, nu_dict, q_dict, q_cent, q_mean, iq_valid



    def _process_q_nu_fit(self, Re_dict, s_dict, nu_dict, q_mean, Nq, nu0, maxfev, rmse):
        

        I_res = np.array([])
        Ierr_res = np.array([])
        q_res = np.array([])

        for iq in tqdm(range(Nq), desc="Processing fitting"):

            if len(np.unique(nu_dict[iq])) > 1:
                _, _, I_fit, I_err = self._leastsq(
                    nu_dict[iq],
                    nu0,
                    Re_dict[iq],
                    s_dict[iq], maxfev, rmse
                )

                I_res = np.append( I_res, I_fit )
                Ierr_res = np.append( Ierr_res, I_err )
                q_res = np.append(q_res, q_mean[iq])

            elif len(np.unique(nu_dict[iq])) == 1:

                # weighted mean
                I_mean = np.average( Re_dict[iq], weights = 1.0 / s_dict[iq]**2 )
                # standard error - the absolute value of s is not given here.
                # this is not precise...
                I_std = np.std(Re_dict[iq], ddof=1) / np.sqrt(len(Re_dict[iq]))

                I_res = np.append( I_res, I_mean )
                Ierr_res = np.append( Ierr_res, I_std )
                q_res = np.append(q_res, q_mean[iq])

        return q_res, I_res, Ierr_res



    def _process_q_nu_fit_2d(self, Re_dict, s_dict, nu_dict, q_dict, q_cen, Nq, nu0, maxfev, rmse):
            

            I_res = np.array([])
            Ierr_res = np.array([])
            q_res = np.array([])

            for iq in tqdm(range(Nq), desc="Processing fitting"):

                dq = (q_dict[iq] - q_cen[iq])/q_cen[iq]

                if len(np.unique(nu_dict[iq])) > 1:
                    I_fit, I_err = self._leastsq_2d(
                        dq,
                        nu_dict[iq],
                        nu0,
                        Re_dict[iq],
                        s_dict[iq], maxfev
                    )

                    I_res = np.append( I_res, I_fit )
                    Ierr_res = np.append( Ierr_res, I_err )
                    q_res = np.append(q_res, q_cen[iq])


            return q_res, I_res, Ierr_res
