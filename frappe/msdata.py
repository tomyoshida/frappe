
import numpy as np

import astropy.constants as cst

import pickle

import os


c = cst.c.cgs.value



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

    def _leastsq(self, nu, nu0, I, sigma):

        x = nu - nu0
        y = I

        # design matrix
        X = np.column_stack((x, np.ones_like(x)))

        # weighted (do not create diagonal matrix)
        Xw = X / sigma[:, None]
        yw = y / sigma

        # coefficient estimation
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        beta = np.linalg.solve(XtX, Xty)

        a, b = beta

        # residuals (original scale)
        r = y - X @ beta

        n, m = X.shape

        # residual variance
        sigma2_hat = np.sum((r / sigma)**2) / (n - m)

        # covariance matrix
        cov = sigma2_hat * np.linalg.inv(XtX)

        # standard deviation
        std_a, std_b = np.sqrt(np.diag(cov))

        return a, std_a, b, std_b  # b is I(nu = nu0)
    

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



    def split(self, vis, dryrun=True):
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

        if dryrun:
            
            return outputvis_arr
            
        else:
            
            # check if working directory exists
            if not os.path.exists(f'./working_{visname}'):
                os.makedirs(f'./working_{visname}')
                print('Created working directory: ' + f'./working_{visname}')
            else:
                # if exists, arise error
                raise FileExistsError(f'Working directory ./working_{visname} already exists. Please remove it before running split.')

            for spw in spw_id:

                outputvis = f'./working_{visname}/spw_id_{spw}.ms'

                os.system('rm -rf ' + outputvis)

                try:
                    self.ct.split(
                        vis = vis,
                        outputvis = outputvis,
                        spw = f'{spw}',
                        keepflags = False,
                        datacolumn = 'CORRECTED_DATA'
                    )
                except:
                    self.ct.split(
                        vis = vis,
                        outputvis = outputvis,
                        spw = f'{spw}',
                        keepflags = False,
                        datacolumn = 'DATA'
                    )
                    
                outputvis_arr.append(outputvis)
        
            return outputvis_arr
        

    def get_visibilities_singlechan(self, vis, pa, incl, FoV, nu0):
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
        
        Re_dict, s_dict, nu_dict, q_dict, Nq = self._binning_q_only(
            q_all, Re_all, sigma2_all, nu_all, FoV
        )
        

        q, V, s = self._process_q_nu_fit(
            Re_dict, s_dict, nu_dict, q_dict, Nq, nu0
        )

        return q, V, s
    

    def get_visibilities(self, vis, nu, pa, incl, FoV, output, save = True):
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

            q[inu], V[inu], s[inu] = self.get_visibilities_singlechan(vis_list, pa, incl, FoV, nu0)

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

        u, v, Re, sigma2, chan_freq, flag = self.load_visibility( vis )

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

        iq_valid = 0

        for iq in range(len(q_cent)):

            mask = q_index == iq

            if mask.any():
                Re_dict[iq_valid] = Re_all[mask]
                s_dict[iq_valid] = np.sqrt(sigma2_all[mask])
                nu_dict[iq_valid] = freq_all[mask]
                q_dict[iq_valid] = q_cent[iq]

                iq_valid += 1

        return Re_dict, s_dict, nu_dict, q_dict, iq_valid



    def _process_q_nu_fit(self, Re_dict, s_dict, nu_dict, q_dict, Nq, nu0):
        

        I_res = np.array([])
        Ierr_res = np.array([])
        q_res = np.array([])

        for iq in range(Nq):
            _, _, I_fit, I_err = leastsq(
                nu_dict[iq],
                nu0,
                Re_dict[iq],
                s_dict[iq]
            )

            I_res = np.append( I_res, I_fit )
            Ierr_res = np.append( Ierr_res, I_err )
            q_res = np.append(q_res, q_dict[iq])

        return q_res, I_res, Ierr_res






