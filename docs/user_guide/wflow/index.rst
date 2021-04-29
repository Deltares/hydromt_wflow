=====
Wflow
=====
With the hydromt_wflow plugin, you can easily work with wflow (SBM) models. This plugin helps you preparing or updating 
several :ref:`components <model_components>` of a wflow model such as topography information, landuse, soil or forcing.
The :ref:`main interactions <model_config>` are available from the HydroMT Command Line Interface and allow you to configure 
HydroMT in order to build or update or clip wflow models. Finally for python users, all wflow objects such as forcing, 
staticmaps etc. are available as :ref:`model attributes <model_attributes>` including some wflow specific ones (on top of the 
`HydroMT model API attributes <https://hydromt.readthedocs.io/en/latest/api/api_model_api.html>`_).

.. toctree::
   :maxdepth: 1

   components
   build_configuration
   attributes
