.. _model_files>:

===============
Wflow datamodel
===============

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowModel` 
attribute contains which Wflow in- and output files. The files are read and written with the associated 
read- and write- methods, i.e. :py:func:`~hydromt_wflow.wflow.WflowModel.read_config` 
and :py:func:`~hydromt_wflow.wflow.WflowModel.write_config` for the 
:py:attr:`~hydromt_wflow.wflow.WflowModel.config`  attribute. 


.. list-table:: WflowModel data
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_wflow.WflowModel` attribute
     - Wflow files
   * - :py:attr:`~hydromt_wflow.WflowModel.config`
     - wflow_sbm.toml
   * - :py:attr:`~hydromt_wflow.WflowModel.staticmaps`
     - staticmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.staticgeoms`
     - geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.)
   * - :py:attr:`~hydromt_wflow.WflowModel.forcing`
     - inmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.states`
     - instates.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.results`
     - output.nc, output_scalar.nc, output.csv
