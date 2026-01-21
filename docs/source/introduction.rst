==============================
Introduction - What is FRAPPE?
==============================

FRAPPE is a Python module for astronomical data analysis. In this page, we'll describe overview of this tool.

Background
==========

Astrophysical disks are amoung the most interesting object in astronomy; they provides unique information for revealing various stages of evolution of astrophysical objects. Amoung them, we are interested in **protoplanetary disks**.

Protoplanetary disks are the birthplace of planetary systems. Planets and other objects form in those disks via coagulation of material such as dust grains. Therefore, in the field of planet formation, it is essential to characterize the proparties of dust grains, for instance, their temperature, spatial distribution, size distribution, as well as mineralogical and chemical compositions. To this end, we routainly perform radio-interferometric observations using e.g., the Atacama Large Millimeter/sub-millimeter Array (ALMA) and obtain the spectral energy distribution (SED) of the dust continuum thermal emission as a function of spatial frequency (a.k.a $(u, v)$)) and electromagnetic frequency.

Such data is the direct observable of the observations and called *interferometric visibility*.
There are mainly two approaches to retrieve the physical quantities that we ultimately want to get;

1. Image-based analysis
    From the visibility data, it is possible to reconstract images (or the intensity distribution on the plane of the sky) by using algorithms such as CLEAN or regularized maximum likelihood techniques. If images at mutiple frequencies are available, we can analyze the SED and constrain physical parameters at each position on the plane of the sky. This is a quite streight forward way. However, imaging algorithms might bring additional uncertainty since the image reconstruction is essentially an inverse problem. Furthermore, this streight-forward method implicitly assumes that the structures are perfectlly resolved, potentially leading biases in the results. To improve the sensitivity, sometimes, azimuthal averaging of the intensities is applied for axisynmetric disks and the radial profiles are instead analyzed. Additionally, the radial intensity profile can be directly derived by a gaussian process approach on the visibility domain ([frankenstein](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract)).

2. Forward modeling
    It is possible to calcurate the visibilities by assuming some parametric physical models. We can further search the parameter sets that reproduce the observations well. This is particularly powerful if the background physics are extremely complex and difficult to directly retrieve the information.

In our opinion, both approaches have pros and cons -- the image-based analysis assumes only minimum physics behind the data but makes difficult to get robust estimates. On the other hand, the forward modeling approach could be biased by the assumption itself.
To reveal the nature of the protoplanetary disks, we need another approach that has a good balance between them. To this end, we have developped a new tool, **FRAPPE**.


What can FRAPPE do?
===================


Features
========

+ Flexible modeling of radial profiles
+ Direct fit to the interferometric visibilities
+ Easy-to-use