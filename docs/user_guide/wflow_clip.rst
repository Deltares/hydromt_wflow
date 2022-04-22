.. _wflow_clip:

Clipping a model
----------------
This plugin allows to clip the following parts of an existing model for a smaller region from command line:

- staticmaps
- forcing

To clip a smaller model from an existing one use:

.. code-block:: console

    activate hydromt-wflow
    hydromt clip wflow path/to/model_to_clip path/to/clipped_model "{'basin' [1001]}" -vvv

As for building, the recommended `region options <https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of the clipped model are:

- basin
- subbasin
   ~WflowModel.setup_precip_forcing
   ~WflowModel.setup_temp_pet_forcing
   ~WflowModel.setup_constant_pars


.. _data: https://deltares.github.io/hydromt/latest/user_guide/data.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options

.. toctree::
    :hidden:

    Example: Clip Wflow model <../_examples/clip_model>
