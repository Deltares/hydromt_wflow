.. currentmodule:: hydromt_wflow

Setup grid from geodataset
==========================

Description
-----------
The ``setup_grid_from_geodataset`` method is part of a set of more generic method to
add data to a Wflow model. This method allows to create a static/cyclic/forcing Wflow
grid from a point geodataset.

The method can for example be used to prepare:

- external inflows: ``river_water__external_inflow_volume_flow_rate``
- reservoir variables if cyclic or forcing and not static, e.g. ``reservoir_water__target_min_volume_fraction``,
  ``reservoir_water__external_inflow_volume_flow_rate`` or ``reservoir_water__outgoing_observed_volume_flow_rate``

The input geodataset can be provided as:

- a netcdf file containing the point locations and the variable to be added. The file
  should contain a `time` and `index` dimensions.
- a csv file containing the static/cyclic/forcing variable values and an additional
  location file. The locations can be provided as a separate csv/vector file or as the
  name of the corresponding layer in the Wflow staticmaps.

Note that the method will not snap the locations to the Wflow grid but will use them as is.
If locations should be snapped, the ``setup_gauges`` method can be used first.

The function can also add the newly created grid to the corresponding wflow variable
in the toml file.

For more details on the arguments and how to use the method, please refer to the API documentation of the
``setup_grid_from_geodataset`` function.

Example usage
-------------
Here are three examples of how to use the ``setup_grid_from_geodataset`` method for a Wflow model:

1. Create static inflows at GRDC gauges. The inflows are given in a csv file and the
  locations using the geosjon file in the wflow staticgeoms folder.
2. Create cyclic reservoir demand at the reservoir outlet locations. The demand is given
    in a csv file and the locations using the `reservoir_outlet_id` map in the staticmaps.
3. Create forcing reservoir outflows at the reservoir outlet locations. The inflows are given
    in a netcdf file containing the variable values and the locations of the reservoir outlets.

1. Static inflows at GRDC gauges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this example, we will add static inflows at the GRDC gauges locations. For the locations, we use the
**gauges_grdc.geosjon** file that is in the staticgeoms folder of the Wflow model. The first
few columns of our geosjon attributes look like this:

.. list-table::
    :header-rows: 1

    * - fid
      - grdc_no
      - station
    * - 0
      - 6349410
      - CANCIA
    * - 1
      - 6349400
      - PONTE DELLA IASTA
    * - 2
      - 6349411
      - PODESTAGNO

The variable values are given in a csv file with the following format. The first line
contains the locations index matching here the `grdc_no` column in the geojson file. The
second line contains the static inflow values in m3/s for each location:

.. list-table::
    :header-rows: 1

    * - time
      - 6349410
      - 6349400
      - 6349411
    * - 0
      - 1.5
      - -1.5
      - 0.5

Now let's use the ``setup_grid_from_geodataset`` method to add these static inflows to our Wflow model.

.. tab-set::

    .. tab-item:: Command Line Interface (CLI)

        The definition of the method and the arguments is done in a workflow file (YAML format).
        The workflow file can then be used to build or update a model from the command line interface.
        Here our input files have a simple format so we can use file paths instead of data
        catalog entries:

        .. code-block:: console

            $ hydromt update wflow_sbm "./path/to/model_to_update" -o "./path/to/model_with_inflows" -i "./path/to/add_inflows.yaml" -v

        The workflow YAML file (``add_inflows.yaml``) would look like this:

        .. code-block:: yaml

            steps:
              - setup_grid_from_geodataset:
                  geodataset_fn: "./path/to/static_inflows.csv" # dataframe with the inflow values
                  locations_fn: "./path/to/model_to_update/staticgeoms/gauges_grdc.geojson" # corresponding locations file
                  index_col: "grdc_no" # column in the geodataset attributes that contains the location index matching the variable file
                  variable: "inflow" # name of the input variable in the geodataset
                  fill_value: 0 # fill value for the other grid cells
                  output_names:
                    "river_water__external_inflow_volume_flow_rate": "river_inflow" # wflow variable and name in staticmaps

    .. tab-item:: Python API

        For python, you need to first instantiate a Wflow model and then call the setup methods directly:

        .. code-block:: python

            from hydromt_wflow import WflowSbmModel

            # instantiate model
            model = WflowSbmModel(
                "./path/to/model_to_update",
                mode="r+",
            )

            # add static inflows at GRDC gauges
            model.setup_grid_from_geodataset(
                geodataset_fn="./path/to/static_inflows.csv", # dataframe with the inflow values
                locations_fn="./path/to/model_to_update/staticgeoms/gauges_grdc.geojson", # corresponding locations file
                index_col="grdc_no", # column in the geodataset attributes that contains the location index matching the variable file
                variable="inflow", # name of the input variable in the geodataset
                fill_value=0, # fill value for the other grid cells
                output_names={"river_water__external_inflow_volume_flow_rate": "river_inflow"} # wflow variable and name in staticmaps
            )

2. Cyclic reservoir demand at reservoir outlet locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this example, we will add cyclic reservoir demand at the reservoir outlet locations.
The locations will be taken using the ``reservoir_outlet_id`` map in the staticmaps.

In this map we have two controlled reservoirs with IDs 3349 and 3367, as well as a natural lake
with ID 169986. We want to add cyclic demand at the two reservoir locations. For the lake,
a fill value of -1 should be used to indicate that this is a natural lake. All other cells
will be masked with a nodata value of -9999.

The demand values in our example are given for the two reservoirs for each month as a csv table.
The first line contains the location index matching the `reservoir_outlet_id` map and the following lines
contain the demand values in m3/s for each month (or day of year, depending on the time resolution of the cyclic variable):

.. list-table::
    :header-rows: 1

    * - time
      - 3349
      - 3367
    * - 1
      - 5.0
      - 0.5
    * - 2
      - 5.0
      - 0.5
    * - 3
      - 7.0
      - 1.0
    * - 4
      - 7.0
      - 1.0
    * - 5
      - 8.0
      - 1.5
    * - 6
      - 8.0
      - 1.5
    * - 7
      - 8.0
      - 1.5
    * - 8
      - 7.0
      - 1.0
    * - 9
      - 7.0
      - 1.0
    * - 10
      - 5.0
      - 0.5
    * - 11
      - 5.0
      - 0.5
    * - 12
      - 5.0
      - 0.5

Now let's use the ``setup_grid_from_geodataset`` method to add these cyclic demands to our Wflow model.

.. tab-set::

    .. tab-item:: Command Line Interface (CLI)

        The workflow YAML file (``add_reservoir_demand.yaml``) would look like this:

        .. code-block:: yaml

            steps:
              - setup_grid_from_geodataset:
                  geodataset_fn: "./path/to/cyclic_reservoir_demand.csv" # dataframe with the demand values
                  locations_fn: "reservoir_outlet_id" # corresponding locations map in staticmaps
                  variable: "reservoir_demand_cyclic" # name of the input variable in the geodataset
                  fill_value: -1 # fill value for the other grid cells (lake)
                  nodata_value: -9999 # nodata values for the cells that are not in the mask
                  mask: "reservoir_outlet_id" # map to use as mask for the output grid (only the cells with a value in this map will be filled, the others will be set to nodata_value)
                  output_names:
                    "reservoir_water_demand__required_downstream_volume_flow_rate": "reservoir_demand_cyclic" # wflow variable and name in staticmaps

    .. tab-item:: Python API

        For python, you need to first instantiate a Wflow model and then call the setup methods directly:

        .. code-block:: python

            from hydromt_wflow import WflowSbmModel

            # instantiate model
            model = WflowSbmModel(
                "./path/to/model_to_update",
                mode="r+",
            )

            # add cyclic reservoir demand at reservoir outlet locations
            model.setup_grid_from_geodataset(
                geodataset_fn="./path/to/cyclic_reservoir_demand.csv", # dataframe with the demand values
                locations_fn="reservoir_outlet_id", # corresponding locations map in staticmaps
                variable="reservoir_demand_cyclic", # name of the input variable in the geodataset
                fill_value=-1, # fill value for the other grid cells (lake)
                nodata_value=-9999, # nodata values for the cells that are not in the mask
                mask="reservoir_outlet_id", # map to use as mask for the output grid (only the cells with a value in this map will be filled, the others will be set to nodata_value)
                output_names={"reservoir_water_demand__required_downstream_volume_flow_rate": "reservoir_demand_cyclic"} # wflow variable and name in staticmaps
            )

3. Forcing reservoir outflows at reservoir outlet locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this example, we will add forcing reservoir outflows at the reservoir outlet locations.
Here both locations and timeseries will come from a netcdf file that we have catalogued.

Here is the main content of the netcdf file:

- data: a 2D array with dimensions `time` and `index` containing the outflow values in m3/s for each reservoir outlet location and time step.
- time: a 1D array containing the time steps of the forcing variable.
- id: a 1D array containing the location index matching the `reservoir_outlet_id` map in the staticmaps.
- lon (coords): a 1D array containing the longitude of the reservoir outlet locations.
- lat (coords): a 1D array containing the latitude of the reservoir outlet locations.

Now let's use the ``setup_grid_from_geodataset`` method to add these forcing outflows to our Wflow model.

.. tab-set::

    .. tab-item:: Command Line Interface (CLI)

        The workflow YAML file (``add_reservoir_outflows.yaml``) would look like this:

        .. code-block:: yaml

            steps:
              - setup_grid_from_geodataset:
                  geodataset_fn: "forcing_reservoir_outflows" # data catalog entry for the netcdf file
                  variable: "outflow" # name of the input variable in the geodataset
                  nodata_value: np.nan # nodata value outside of the mask
                  mask: "reservoir_outlet_id" # map to use as mask for the output grid
                  output_names:
                    "reservoir_water__outgoing_observed_volume_flow_rate": "reservoir_outflow" # wflow variable and name in staticmaps
                  resample_time_kwargs: # time resampling arguments if the time steps of the input variable do not match the model time steps (e.g. hourly vs daily)
                    downsampling: "mean"

    .. tab-item:: Python API

        For python, you need to first instantiate a Wflow model and then call the setup methods directly:

        .. code-block:: python

            from hydromt_wflow import WflowSbmModel

            # instantiate model
            model = WflowSbmModel(
                "./path/to/model_to_update",
                mode="r+",
                data_libs=["my_data_catalog"] # make sure to include the data library where the netcdf file is catalogued
            )

            # add forcing reservoir outflows at reservoir outlet locations
            model.setup_grid_from_geodataset(
                geodataset_fn="forcing_reservoir_outflows", # data catalog entry for the netcdf file
                variable="outflow", # name of the input variable in the geodataset
                nodata_value=np.nan, # nodata value outside of the mask
                mask="reservoir_outlet_id", # map to use as mask for the output grid
                output_names={"reservoir_water__outgoing_observed_volume_flow_rate": "reservoir_outflow"} # wflow variable and name in staticmaps
                resample_time_kwargs={ # time resampling arguments if the time steps of the input variable do not match the model time steps (e.g. hourly vs daily)
                    "downsampling": "mean"
                }
            )
