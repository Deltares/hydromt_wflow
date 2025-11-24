.. currentmodule:: hydromt_wflow

Setup_lulcmaps and related methods
==================================

Description
-----------

To prepare land use / land cover related maps for Wflow, HydroMT provides several
methods. The basis of these methods is that they use lookup tables with parameters
values for each land use / land cover class and then map these values to the model grid
using land use / land cover maps.

The parameters are mapped at the original LULC map resolution before being resampled to
the model grid resolution (using either averaging or majority mapping depending on the
parameter type). This ensures that most of the details of the original resolution of the
LULC map are preserved in the final model maps.

.. figure:: _static/setup_lulcmaps.png

The following methods are available:

- :py:meth:`~WflowSbmModel.setup_lulcmaps`: Main method to setup LULC maps using lookup tables.
- :py:meth:`~WflowSbmModel.setup_lulcmaps_from_vector`: Similar to the above but starts with
  rasterizing the LULC vector data to a user-defined resolution.
- :py:meth:`~WflowSbmModel.setup_lulcmaps_with_paddy`: Specific method if paddies are present
  in your catchment. The LULC map can directly contain a paddy class or an additional paddy map
  can be provided and will be merged into the landuse map before deriving parameters. Additional
  parameters related to paddy management (minimum/optimal/maximum water levels) are also added
  based on user defined values. Finally, to allow for water to pool on the surface
  (for paddy/rice fields), the layers in the model can be updated to new depths, such that we
  can allow a thin layer with limited vertical conductivity. These updated layers means that the
  ``soil_brooks_corey_c`` parameter needs to be calculated again. Next, the
  soil_ksat_vertical_factor layer corrects the vertical conductivity
  (by multiplying) such that the bottom of the layer corresponds to a
  `target_conductivity` for that layer.


Parameter lookup tables
-----------------------
The parameter values for each land use / land cover class need to be defined in lookup tables.
HydroMT provides some default lookup tables for the following LULC classification systems:

- **corine**: `CORINE Land Cover (CLC) <https://land.copernicus.eu/en/products/corine-land-cover>`_
- **glcnmo**: `GLCNMO <https://globalmaps.github.io/glcnmo.html>`_
- **globcover**: `GlobCover <https://due.esrin.esa.int/page_globcover.php>`_
- **esa_worldcover**: `ESA WorldCover <https://esa-worldcover.org/en>`_
- **vito**: `Copernicus Global Dynamic Land Cover <https://land.copernicus.eu/en/products/global-dynamic-land-cover>`_
- **paddy**: specific lookup table for paddy fields (if using the ``setup_lulcmaps_with_paddy`` method)

You can find these tables in the `HydroMT-Wflow repository <https://github.com/Deltares/hydromt_wflow/tree/main/hydromt_wflow/data/lulc>`_
or create your own based on these examples or literature values to better reflect the specific vegetation
and soil characteristics of your study area.

The lookup tables are simple CSV files with the first column containing the land use / land cover class
identifiers (matching those in the LULC map), the second column containing the ``description`` of the class
and the other columns containing the parameter values for each class. The **last line of the table should
contain the nodata value** for the LULC map (e.g. -9999) and the corresponding parameter values for nodata areas.

The columns names should match the HydroMT names of each Wflow parameter. These are:

- **landuse**: landuse class ID
- **vegetation_kext**: Extinction coefficient in the canopy gap fraction equation [-]
- **land_manning_n**: Manning Roughness [m-1/3 s]
- **soil_compacted_fraction**: The fraction of compacted or urban area per grid cell [-]
- **vegetation_root_depth**: Length of vegetation roots [mm]
- **vegetation_leaf_storage**: Specific leaf storage [mm]
- **vegetation_wood_storage**: Fraction of wood in the vegetation/plant [-]
- **land_water_fraction**: The fraction of open water per grid cell [-]
- **vegetation_crop_factor**: Crop coefficient [-]
- **vegetation_feddes_alpha_h1**: Root water uptake reduction at soil water pressure head h1 (0 or 1) [-]
- **vegetation_feddes_h1**: Soil water pressure head h1 at which root water uptake is reduced (Feddes) [cm]
- **vegetation_feddes_h2**: Soil water pressure head h2 at which root water uptake is reduced (Feddes) [cm]
- **vegetation_feddes_h3_high**: Soil water pressure head h3 (high) at which root water uptake is reduced (Feddes) [cm]
- **vegetation_feddes_h3_low**: Soil water pressure head h3 (low) at which root water uptake is reduced (Feddes) [cm]
- **vegetation_feddes_h4**: Soil water pressure head h4 at which root water uptake is reduced (Feddes) [cm]
- **erosion_usle_c** (sediment): USLE cover management factor [-]

Example lookup table (for ESA WorldCover):
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO update after harmo is done!!!!!!!!!!!!!!!!!!!

.. code-block:: csv

    esa,description,landuse,vegetation_kext,land_manning_n,soil_compacted_fraction,vegetation_root_depth,vegetation_leaf_storage,vegetation_wood_storage,land_water_fraction,vegetation_crop_factor,vegetation_feddes_alpha_h1,vegetation_feddes_h1,vegetation_feddes_h2,vegetation_feddes_h3_high,vegetation_feddes_h3_low,vegetation_feddes_h4,erosion_usle_c,Cov_River
    10,Tree cover,10,0.8,0.5,0,369,0.0477,0.5,0,0.85,1,0,-100,-400,-1000,-15849,0.0069,12.3
    20,Shrubland,20,0.07,0.5,0,410,0.07,0.1,0,0.8,1,0,-100,-400,-1000,-15849,0.05,1.97
    30,Grassland,30,0.6,0.15,0,106.8,0.1272,0,0,0.75,1,0,-100,-400,-1000,-15849,0.045,1.97
    40,Cropland,40,0.6,0.2,0,390.4,0.1272,0,0,1.15,0,0,-100,-400,-1000,-15849,0.3,1.97
    50,Built-up,50,0.7,0.011,0.9,257.4,0.04,0.01,0,1,1,0,-100,-400,-1000,-15849,0,1
    60,Bare / sparse vegetation,60,0.6,0.02,0,10.7,0.04,0.04,0,1,1,0,-100,-400,-1000,-15849,0.25,1.97
    70,Snow and Ice,70,0.6,0.01,0,0,0,0,0,1,1,0,-100,-400,-1000,-15849,0,1
    80,Permanent water bodies,80,0.7,0.01,0,0,0,0,1,1,1,0,-100,-400,-1000,-15849,0,1
    90,Herbaceous wetland,90,0.6,0.15,0,106.8,0.1272,0,0,1.1,1,0,-100,-400,-1000,-15849,0.05,1.97
    95,Mangroves,95,0.8,0.5,0,369,0.0477,0.5,0.5,1,1,0,-100,-400,-1000,-15849,0.0069,12.3
    100,Moss and lichen,100,0.6,0.085,0,136.9,0.04,0,0,1,1,0,-100,-400,-1000,-15849,0.04,1.97
    0,No data,0,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999

Parameter estimation
--------------------
The estimates in the above table are based on literature reviews done by
`Imhoff et al., 2020 < https://doi.org/10.1029/2019WR026807>`_

Here are some references to help you estimate parameter values for your own lookup tables.

Interception parameters
^^^^^^^^^^^^^^^^^^^^^^^
Parameters related to vegetation interception and storage of rainfall on leaves and branches.

+--------------+----------------------------------------------------------------+----------+-----------------------------------------------------------------------------------+
| Parameter    | Description                                                    | Range    | Reference                                                                         |
+==============+================================================================+==========+===================================================================================+
| kext         | Extinction coefficient in the canopy gap fraction equation [-] | 0.2-0.9  | `Van Dijk and Bruijnzeel (2001) <https://doi.org/10.1016/S0022-1694(01)00392-4>`_ |
|              |                                                                |          | `Van Heemst (1988) <https://edepot.wur.nl/218353>`_                               |
+--------------+----------------------------------------------------------------+----------+-----------------------------------------------------------------------------------+
| leaf_storage | Specific leaf storage [mm]                                     | 0.02-0.2 | `Pitman (1989) <https://doi.org/10.5194/hess-15-3355-2011>`_                      |
|              |                                                                |          | `Liu (1998) <https://doi.org/10.1016/S0022-1694(98)00115-2>`_                     |
+--------------+----------------------------------------------------------------+----------+-----------------------------------------------------------------------------------+
| wood_storage | Fraction of wood in the vegetation/plant [-]                   | 0.0-0.5  | `Pitman (1989) <https://doi.org/10.5194/hess-15-3355-2011>`_                      |
|              |                                                                |          | `Liu (1998) <https://doi.org/10.1016/S0022-1694(98)00115-2>`_                     |
+--------------+----------------------------------------------------------------+----------+-----------------------------------------------------------------------------------+

**kext**

Extract from Van Dijk and Bruijnzeel (2001):

   The value of kext for a particular radiation wavelength depends on leaf
   distribution and inclination angle and for PAR usually ranges between 0.6
   and 0.8 in forests (Ross, 1975). For a number of agricultural crops,
   van Heemst (1988) reported kext values between 0.2 and 0.8 with values of
   0.5-0.7 being the most common.


Values for different crops from van Heemst (1988):

.. list-table::
   :header-rows: 1

   * - Crop
     - kext
   * - Wheat
     - 0.42 - 0.54
   * - Barley
     - 0.44
   * - Rice
     - 0.29 - 0.43
   * - Millet
     - 0.5 - 0.6
   * - Sorghum
     - 0.4 - 0.7
   * - Maize
     - 0.6 - 0.64
   * - Soybean
     - 0.787 - 0.804
   * - Peanut
     - 0.6
   * - Oilseed rape
     - 0.54
   * - Sunflower
     - 0.8 - 0.9
   * - Cassava
     - 0.7 - 0.88
   * - Sweet Potato
     - 0.45
   * - Potato
     - 0.48
   * - Sugar beet
     - 0.65
   * - Sugar cane
     - 0.48
   * - Cotton
     - 0.62

Evapotranspiration parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters related to vegetation evaporation and transpiration.

+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| Parameter       | Description                                         | Range                           | Reference                                                                                     |
+=================+=====================================================+=================================+===============================================================================================+
| crop_factor     | Crop coefficient [-]                                |                                 |                                                                                               |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| root_depth      | Length of vegetation roots [mm]                     | 100 - 5000                      | `Fan et al. (2016) <https://www.mdpi.com/2077-0472/14/4/532>`_                                |
|                 |                                                     |                                 | `Schenk and Jackson (2002) <https://doi.org/10.1890/0012-9615(2002)072[0311:TGBOR]2.0.CO;2>`_ |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_alpha_h1 | Root water uptake reduction at pressure head h1 [-] | 0 (crop) - 1 (other)            | `Feddes et al. (1978) <https://edepot.wur.nl/172222>`_                                        |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h1       | Critical pressure head h1 - anorexic condition [cm] | 100 (paddy) - 0 (other)         | `Feddes et al. (1978) <https://edepot.wur.nl/172222>`_                                        |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h2       | Critical pressure head h2 - field capacity [cm]     | 55 (paddy) - -100 (other)       | `Feddes et al. (1978) <https://edepot.wur.nl/172222>`_                                        |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h3_high  | Critical pressure head h3 (high) [cm]               | -160 (paddy) - -400 (other)     | `Feddes et al. (1978) <https://edepot.wur.nl/172222>`_                                        |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h3_low   | Critical pressure head h3 (low) [cm]                | -250 (paddy) - -1000 (other)    | `Feddes et al. (1978) <https://edepot.wur.nl/172222>`_                                        |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h4       | Critical pressure head h4 - wilting point [cm]      | -15000 (paddy) - -15849 (other) | `Feddes et al. (1978) <https://edepot.wur.nl/172222>`_                                        |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+

**Crop factor**

- https://www.fao.org/4/x0490e/x0490e0b.htm#chapter%206%20%20%20etc%20%20%20single%20crop%20coefficient%20(kc)

**Root depth**

Values for different crops from Fan et al. (2016) and different other vegetation from Schenk and Jackson (2002):

.. list-table::
   :header-rows: 1

   * - Vegetation / Crop type
     - Root depth D50 [mm]
     - Root depth D95 [mm]
   * - Tundra
     - 90
     - 290
   * - Boreal forest
     - 120
     - 580
   * - Cool temperate forest
     - 210
     - 1040
   * - Warm temperate forest
     - 230
     - 1210
   * - Meadows
     - 50
     - 400
   * - Prairie
     - 70
     - 910
   * - Semi arid steppe
     - 160
     - 1200
   * - Temperate savanna
     - 230
     - 1400
   * - Mediterranean woodland and shrub
     - 190
     - 1710
   * - Semi-desert shrubland
     - 280
     - 1310
   * - Desert
     - 270
     - 1120
   * - Dry tropical savannas
     - 280
     - 1440
   * - Humid tropical savannas
     - 140
     - 940
   * - Tropical semi-deciduous and deciduous forest
     - 160
     - 950
   * - Tropical evergreen forest
     - 150
     - 910
   * - Wheat
     - 168
     - 1038
   * - Maize
     - 144
     - 889
   * - Oat
     - 112
     - 777
   * - Barley
     - 115
     - 996
   * - Cereals
     - 141
     - 929
   * - Soybean
     - 109
     - 1380
   * - Oilseed crops
     - 94
     - 1063
   * - All crops
     - 146
     - 1027

**Feddes root water uptake**

Other links:

- https://books.google.com.sg/books?hl=fr&lr=&id=e0MzVX-7FnIC&oi=fnd&pg=PA95&dq=Parameterizing+the+soil+%E2%80%93+water+%E2%80%93+plant+root+system+R.A.+Feddes%23+and+P.A.C.+Raats&ots=ZfjyLAP3CN&sig=WIAfKWDerr6rKwb2-gDNX4KQPhk&redir_esc=y#v=onepage&q=Parameterizing%20the%20soil%20%E2%80%93%20water%20%E2%80%93%20plant%20root%20system%20R.A.%20Feddes%23%20and%20P.A.C.%20Raats&f=false
- https://swap.wur.nl/DownloadHistory/swap303/Reference%20Manual%20SWAP%20version%203.0.3%20Report773.pdf (appendix 3)
- https://www.mdpi.com/2077-0472/14/4/532

Manning Roughness
^^^^^^^^^^^^^^^^^
Manning's N values are used to represent roughness of the land surface for overland flow.
Estimations per landuse class can be found in literature such as:

+-----------+-----------------------------+-------------+--------------------------------------------------------------------------+
| Parameter | Description                 | Range       | Reference                                                                |
+===========+=============================+=============+==========================================================================+
| manning_n | Manning Roughness [m-1/3 s] | 0.008-0.96  | `Engman (1986) <https://doi.org/10.1061/(ASCE)0733-9437(1986)112:1(39)>` |
|           |                             |             | `Kilgore (1997) <http://hdl.handle.net/10919/35777>`_                    |
|           |                             |             | Cronshey (1986)                                                          |
+-----------+-----------------------------+-------------+--------------------------------------------------------------------------+

Example of values from different sources:

.. list-table::
   :header-rows: 1

   * - Landuse
     - Cronshey
     - Kilgore
     - Engman
   * - Smooth surfaces (concrete, gravel, bare)
     - 0.011
     - 0.015 (residential/commercial) / 0.020 (gravel road)
     - 0.010 (bare) / 0.011 (concrete) / 0.020 (gravel)
   * - Fallow (no residue)
     - 0.05
     - 0.05
     -
   * - Cultivated soil
     - 0.06 - 0.17 (depending on residue cover)
     - 0.032 (wheat) - 0.08 (corn) - 0.2 (depending on tillage)
     -
   * - Grassland
     - 0.15 (short) / 0.24 (dense)
     - 0.046 (grass) / 0.1 (pasture)
     - 0.15 (short grass)
   * - Forest
     - 0.4 - 0.8 (depending on underbrush)
     - 0.6
     - 0.3 - 0.8
   * - Range (natural)
     - 0.13
     -
     - 0.13
   * - Wetland
     -
     - 0.125
     -
   * - Waterway, pond
     -
     - 0.08
     -

Land cover parameters
^^^^^^^^^^^^^^^^^^^^^
The land cover parameters represent fractions of different land cover types within each grid cell.
For these parameters, it may matter to take into account the resolution of the original LULC map
when estimating the values. E.g. a coarse resolution may for example represent a cell that is majority
urban but still contains a significant fraction of vegetation or water.

+-------------------------+-------------------------------------------------------+---------------------------------------------------------------+
| Parameter               | Description                                           | Estimate                                                      |
+=========================+=======================================================+===============================================================+
| soil_compacted_fraction | Fraction of compacted or paved area per grid cell [-] | > 0 if urban or compacted / paved surfaces are present        |
+-------------------------+-------------------------------------------------------+---------------------------------------------------------------+
| land_water_fraction     | Fraction of open water per grid cell [-]              | > 0 if water (water bodies, ponds, waterways etc.) is present |
+-------------------------+-------------------------------------------------------+---------------------------------------------------------------+

Soil erosion
^^^^^^^^^^^^
USLE cover management factor (erosion_usle_c)

References
----------

- Cronshey, R. (1986). Urban hydrology for small watersheds (No. 55). US Department of Agriculture,
  Soil Conservation Service, Engineering Division.
- van Dijk, A. I. J. M., & Bruijnzeel, L. A. (2001). Modelling rainfall interception by vegetation
  of variable density using an adapted analytical model. Part 2. Model validation for a tropical
  upland mixed cropping system. Journal of Hydrology, 247(3-4), 239–262.
- Engman, E. (1986). Roughness coefficients for routing surface runoff. Journal of Irrigation
  and Drainage Engineering, 112(1), 39-53. https://doi.org/10.1061/(ASCE)0733-9437(1986)112:1(39)
- Fan, J., McConkey, B., Wang, H., & Janzen, H. (2016). Root distribution by depth for temperate
  agricultural crops. Field Crops Research, 189, 68–74. https://doi.org/10.1016/j.fcr.2016.02.013
- Feddes, R.A., Kowalik, P.J. and Zaradny, H., 1978, Simulation of field water use and crop yield,
  Pudoc, Wageningen, Simulation Monographs.
- van Heemst, H.D.J. (1988). Plant data values required for simple crop growth simulation models,
  review and bibliography. Simulation report CABO-TT No 17. Wageningen.
- Imhoff, R.O, van Verseveld, W.J., van Osnabrugge, B., Weerts, A.H., 2020. Scaling Point-Scale
  (Pedo)transfer Functions to Seamless Large-Domain Parameter Estimates for High-Resolution
  Distributed Hydrologic Modeling: An Example for the Rhine River. Water Resources Research,
  56, e2019WR026807. https://doi.org/10.1029/2019WR026807.
- Kilgore, J. L. (1997). Development and evaluation of a GIS-based spatially distributed unit
  hydrograph model (MSc thesis). Retrieved from http://hdl.handle.net/10919/35777
- Liu, S. (1998). Estimation of rainfall storage capacity in the canopies of cypress wet lands
  and slash pine uplands in North-Central Florida. Journal of Hydrology, 207(1-2), 32–41.
  https://doi.org/10.1016/S0022-1694(98)00115-2
- Pitman, J. (1989). Rainfall interception by bracken in open habitats—Relations between
  leaf area, canopy storage and drainage rate. Journal of Hydrology, 105(3-4), 317–334.
  https://doi.org/10.1016/0022-1694(89)90111-X
- Schenk, H. J., & Jackson, R. B. (2002). The global biogeography of roots. Ecological
  Monographs, 72(3), 311–328. https://doi.org/10.1890/0012-9615(2002)072[0311:TGBOR]2.0.CO;2
