==============================
Introduction - What is FRAPPE?
==============================

FRAPPE is a Python module for astronomical data analysis. In this page, we describe an overview of this tool.

Background
==========

Astrophysical disks are among the most interesting objects in astronomy; they provide unique information for revealing various stages of evolution of astrophysical objects. Among them, we are interested in **protoplanetary disks**.

Protoplanetary disks are the birthplace of planetary systems. Planets and other objects form in those disks via coagulation of material such as dust grains. Therefore, in the field of planet formation, it is essential to characterize the properties of dust grains, for instance, their temperature, spatial distribution, size distribution, as well as mineralogical and chemical compositions. To this end, we routinely perform radio-interferometric observations using e.g., the Atacama Large Millimeter/sub-millimeter Array (ALMA) and obtain the spectral energy distribution (SED) of the dust continuum thermal emission as a function of spatial frequency (a.k.a (u, v)) and electromagnetic frequency.

Such data are the direct observables from the observations and are called *interferometric visibilities*.
There are mainly two approaches to retrieve the physical quantities that we ultimately want to obtain:

1. Image-based analysis
    From the visibility data, it is possible to reconstruct images (or the intensity distribution on the plane of the sky) by using algorithms such as CLEAN or regularized maximum likelihood techniques. If images at multiple frequencies are available, we can analyze the SED and constrain physical parameters at each position on the plane of the sky. This is quite a straightforward way. However, imaging algorithms might bring additional uncertainty since the image reconstruction is essentially an inverse problem. Furthermore, this straightforward method implicitly assumes that the structures are perfectly resolved, potentially leading to biases in the results. To improve the sensitivity, sometimes, azimuthal averaging of the intensities is applied for axisymmetric disks and the radial profiles are instead analyzed. The radial intensity profile can also be directly derived by a Gaussian process approach on the visibility domain (`frankenstein <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract>`_).

2. Forward modeling
    It is possible to calculate the visibilities by assuming some parametric physical models. We can further search for the parameter sets that reproduce the observations well. This is particularly powerful if the background physics is extremely complex and it is difficult to directly retrieve information.

In our opinion, both approaches have pros and cons -- the image-based analysis assumes only minimal physics behind the data but makes it difficult to obtain robust estimates. On the other hand, the forward modeling approach could be biased by the assumptions themselves.
To reveal the nature of protoplanetary disks, we need another approach that strikes a good balance between them. To this end, we have developed a new tool, **FRAPPE**.


What can FRAPPE do?
===================

+ Flexible modeling of radial profiles
+ Direct fit to the interferometric visibilities
+ Easy-to-use