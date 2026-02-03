==============================
Introduction - What is FRAPPE?
==============================

FRAPPE is a Python module for astronomical data analysis. On this page, we provide an overview of the tool.

Background
==========

**Protoplanetary disks** are the birthplaces of planetary systems. Planets and other bodies form in these disks via the coagulation of material such as dust grains. Consequently, in planet-formation studies it is essential to characterize dust-grain properties—such as temperature, spatial distribution, size distribution, and mineralogical or chemical composition. To this end, we routinely perform radio-interferometric observations with, for example, the Atacama Large Millimeter/submillimeter Array (ALMA) and obtain the spectral energy distribution (SED) of dust thermal continuum emission as a function of spatial frequency (i.e., (u, v)) and electromagnetic frequency.

These data are the direct observables from the measurements and are called *interferometric visibilities*.
There are two main approaches to retrieve the physical quantities of interest:

1. Image-based analysis
    From the visibility data, we can reconstruct images (the intensity distribution on the sky) using algorithms such as CLEAN or regularized maximum-likelihood techniques. If images at multiple frequencies are available, we can analyze the SED and constrain physical parameters at each position on the sky plane. This approach is straightforward, but imaging algorithms can introduce additional uncertainty because image reconstruction is fundamentally an underdetermined problem. Furthermore, if spatial resolutions differ among frequencies, we typically have to convolve images to the largest common beam size for a fair comparison, losing the resolution. This method also implicitly assumes that structures are perfectly resolved, which can bias the results. To improve sensitivity, azimuthal averaging of intensities is sometimes applied to axisymmetric disks, and the radial profiles at each frequency are analyzed instead. The radial intensity profile can also be derived directly via a Gaussian-process approach in the visibility domain (`frankenstein <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract>`_).

2. Forward modeling
    We can calculate the visibilities by assuming parametric physical models. We can then search for parameter sets that best reproduce the observations. This approach is particularly powerful when the underlying physics is extremely complex and direct retrieval is difficult.

Both approaches have pros and cons: image-based analysis assumes minimal physics but can make robust estimation difficult. Conversely, forward modeling can be biased by its own assumptions.
To reveal the nature of protoplanetary disks, we need an approach that balances these two methods. To this end, we have developed **FRAPPE**.


What can FRAPPE do?
===================

With FRAPPE, you can directly retrieve dust properties from interferometric visibilities of protoplanetary disks. Its features can be summarized as follows:

+ Visibility-based analysis
    FRAPPE can process visibilities directly. You no longer need to worry about differences in beam size. This is especially advantageous when using lower frequencies, where better spatial resolution is more challenging. It also reduces uncertainties introduced by imaging.

+ Flexible radial profiles with a Gaussian Process
    FRAPPE assumes that the underlying physical parameters can be expressed as a sample path from a `Gaussian process <https://en.wikipedia.org/wiki/Gaussian_process>`_. This enables highly flexible modeling that is impossible with the forward-modeling approaches. 

+ Auto-differentiation with JAX
    We implemented almost all functions using the `JAX <https://docs.jax.dev/en/latest/index.html>`_ library. This makes end-to-end automatic differentiation possible. For example, each visibility datum can be automatically and analytically differentiated with respect to each physical parameter. This enables the use of modern Markov chain Monte Carlo (MCMC) methods such as Hamiltonian Monte Carlo. It allows sampling of highly correlated, high-dimensional parameter spaces using gradient information, which is not feasible with traditional MCMC.

+ Bayesian inference with numpyro
    MCMC sampling and stochastic variational inference are implemented using the `numpyro <https://num.pyro.ai/en/stable/>`_ package, providing extensibility. 

+ Easy to use
    FRAPPE is designed to be easy to use in day-to-day research. A few simple lines of Python are enough to obtain results.


Tutorials
=========

.. toctree::
   :maxdepth: 2

   tutorial_0
   tutorial_1


What's this logo?
=================

The logo shown in the top left is `Shirokuma <https://en.wikipedia.org/wiki/Kakig%C5%8Dri#Shirokuma>`_, a type of frappe (more precisely, a shaved ice dessert) originated from Tenmonkan (which means an *astronomical observatory* in the Edo period), `Kagoshima <https://en.wikipedia.org/wiki/Kagoshima>`_, Japan.

