==============
Wflow Sediment
==============

Extending a wflow model with a wflow_sediment model
---------------------------------------------------
If you already have a wflow model and you want to extend it in order to include sediment as well, then you do not need to build the
wflow_sediment from scratch. You can instead ``update`` the wflow model with the additional components needed by wflow_sediment.
These components are available in a template :download:`.ini file <../_examples/wflow_extend_sediment.ini>` and shown below. The corresponding
command line would be:

.. code-block:: console

    activate hydromt-wflow
    hydromt update wflow_sediment path/to/wflow_model_to_extend -i wflow_extend_sediment.ini -d data_sources.yml -vvv

.. literalinclude:: ../_examples/wflow_extend_sediment.ini
   :language: Ini

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options


With the hydromt_wflow plugin, you can easily work with wflow sediment models.
This plugin contains as well relevant functions for setting up or adjusting wflow sediment models:

* :ref:`building a model <sediment_build>`: building a model from scratch.
* :ref:`updating a model <sediment_update>`: updating an existing model (e.g. update datafeeds).
* :ref:`clipping a model <_sediment_clip>`: changing the spatial domain of an existing model (e.g. select subbasins from a larger model).


.. toctree::
    :maxdepth: 2
    :hidden:

    sediment_model_setup.rst
    sediment_build.rst
    sediment_clip.rst
    sediment_update.rst
