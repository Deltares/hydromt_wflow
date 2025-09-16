.. currentmodule:: hydromt_wflow
.. _wflow_clip:

Clipping a model
----------------
This plugin allows to clip the following parts of an existing model for a smaller region from command line:

- staticmaps
- forcing
- states
- geoms
- config (update reservoir settings)
- tables (update reservoir rating curves)

To clip a smaller model from an existing one use the ``update`` CLI command with the **clip** method:

.. code-block:: console

    activate hydromt_wflow
    hydromt update wflow_sbm path/to/model_to_clip -o path/to/clipped_model -i clip_config.yml -v

Here is an example of the clip config:

.. code-block:: yaml

    steps:
      - clip:
        region: {"basin": [x, y]} # region to clip the model too
        inverse_clip: false # whether to clip outside or inside the region
        clip_states: true # whether to clip states
        clip_forcing: true # whether to clip forcing

As for building, the recommended `region options <https://deltares.github.io/hydromt/stable/guides/user_guide/model_region.html>`_
for a proper implementation of the clipped model are:

- basin
- subbasin

See the following model API:

* :py:func:`~WflowSbmModel.clip`


.. .. toctree::
    :hidden:

    Example Clip Wflow model <../_examples/clip_model.ipynb>
