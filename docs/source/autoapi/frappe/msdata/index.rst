frappe.msdata
=============

.. py:module:: frappe.msdata


Attributes
----------

.. autoapisummary::

   frappe.msdata.c


Classes
-------

.. autoapisummary::

   frappe.msdata.ms


Module Contents
---------------

.. py:data:: c

.. py:class:: ms

   .. py:method:: split(vis, dryrun=True)

      split the measurement set by spectral windows.

      :param vis: Path to the measurement set.
      :type vis: str
      :param dryrun: If True, only simulate the split without executing it. Defaults to True.
      :type dryrun: bool, optional

      :returns: List of paths to the split measurement sets.
      :rtype: list



   .. py:method:: get_visibilities_singlechan(vis, pa, incl, FoV, nu0)

      get one-dimensional deprojected visibilities and uncertainty as a function of uv-distance for a single measurement sets list.

      This function processes multiple measurement sets, deprojects the visibilities based on the provided position angle and inclination, azimuthally averages the data according to the specified field of view, and fits the visibilities by a linear function to extract intensity values and their uncertainties at a reference frequency.

      :param vis: List of paths to the measurement sets.
      :type vis: list
      :param pa: Position angle in degrees for deprojection.
      :type pa: float
      :param incl: Inclination angle in degrees for deprojection.
      :type incl: float
      :param FoV: Field of view in arcseconds to determine the bin size for azimuthal averaging.
      :type FoV: float
      :param nu0: Reference frequency in Hz for fitting the visibilities.
      :type nu0: float

      :returns:

                A tuple containing:
                    - q (np.ndarray): Deprojected uv-distances in 1/radians.
                    - V (np.ndarray): Fitted intensity values at the reference frequency.
                    - s (np.ndarray): Uncertainties associated with the fitted intensity values.
      :rtype: tuple



   .. py:method:: get_visibilities(vis, nu, pa, incl, FoV, output, save=True)

      get one-dimensional deprojected visibilities and uncertainty as a function of uv-distance for multiple measurement set lists

      This function processes multiple sets of measurement sets, deprojects the visibilities based on the provided position angle and inclination, azimuthally averages the data according to the specified field of view, and fits the visibilities by a linear function to extract intensity values and their uncertainties at given reference frequencies.

      :param vis: List of lists, where each sublist contains paths to measurement sets for a specific frequency channel.
      :type vis: list
      :param nu: List of reference frequencies in Hz corresponding to each set of measurement sets.
      :type nu: list
      :param pa: Position angle in degrees for deprojection.
      :type pa: float
      :param incl: Inclination angle in degrees for deprojection.
      :type incl: float
      :param FoV: Field of view in arcseconds to determine the bin size for azimuthal averaging.
      :type FoV: float
      :param output: Base path for saving the output pickle file.
      :type output: str
      :param save: If True, saves the output data to a pickle file. Defaults to True.
      :type save: bool, optional



