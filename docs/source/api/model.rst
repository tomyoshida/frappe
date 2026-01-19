Model API
=========

The ``frappe.model`` module hosts the main Gaussian-process visibility model, observation container, inference helpers, and plotting utilities used throughout FRAPPE.

Quickstart
----------

.. code-block:: python

   from frappe.model import model, inference

   # Build a model and register free parameters
   disk = model(incl=45.0, r_out=1.0, N_GP=64)
   disk.set_parameter('log10_Sigma_d', bounds=(-3, 3), mean_std=(0.0, 0.5), GP=True)
   disk.set_parameter('log10_T', bounds=(1, 3), mean_std=(2.0, 0.2), GP=True)
   disk.set_parameter('q', free=True, GP=False, bounds=(0.0, 4.0), mean_std=(1.0, 0.3))
   disk.set_parameter('log10_a_max', free=True, GP=False, bounds=(-1, 4), mean_std=(1.5, 0.5))

   # Attach observations and opacity tables (arrays omitted here)
   # disk.set_observations(band='B6', q=q_vals, V=vis, s=vis_err, s_f=0.1, mean_fs=1.0, nu=nu_vals, Nch=len(nu_vals))
   # disk.set_opacity(opac_dict)

   # Run inference
   infer = inference(disk)
   prior_draws = infer.prior(num_samples=16)
   # svi_map = infer.SVI_MAP(num_iterations=1000)
   # mcmc = infer.MCMC(num_warmup=500, num_samples=1000, step_size=0.1)

Reference
---------

.. automodule:: frappe.model
   :members: model, observation, inference, results, plot
   :undoc-members:
   :show-inheritance:
