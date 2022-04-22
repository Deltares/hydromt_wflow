.. _model_files_sed:

WflowSediment datamodel
=======================

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowSedimentModel` 
attribute contains which WflowSediment in- and output files. The files are read and written with the associated 
read- and write- methods, i.e. :py:func:`~hydromt_wflow.WflowSedimentModel.read_config` 
and :py:func:`~hydromt_wflow.WflowSedimentModel.write_config` for the 
:py:attr:`~hydromt_wflow.WflowSedimentModel.config`  attribute. 


.. list-table:: WflowSedimentModel data
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_wflow.WflowSedimentModel` attribute
     - Wflow sediment files
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.config`
     - wflow_sediment.toml
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.staticmaps`
     - staticmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.staticgeoms`
     - geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.)
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.forcing`
     - inmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.states`
     - instates.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.results`
     - output.nc, output_scalar.nc, output.csv
