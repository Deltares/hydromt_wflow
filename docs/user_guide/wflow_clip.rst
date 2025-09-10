.. currentmodule:: hydromt_wflow
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

As for building, the recommended `region options <https://deltares.github.io/hydromt/stable/guides/user_guide/model_region.html>`_
for a proper implementation of the clipped model are:

- basin
- subbasin

See the following model API:

* :py:func:`~WflowSbmModel.clip_grid`
* :py:func:`~WflowSbmModel.clip_forcing`
* :py:func:`~WflowSbmModel.clip_states`
* :py:func:`~WflowSedimentModel.clip_grid`
* :py:func:`~WflowSedimentModel.clip_forcing`
* :py:func:`~WflowSedimentModel.clip_states`


.. .. toctree::
    .. :hidden:

    .. Example: Clip Wflow model <../_examples/clip_model.ipynb>
