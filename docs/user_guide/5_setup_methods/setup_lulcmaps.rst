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

- :py:meth:`~WflowSbmModel.setup_lulcmaps` and :py:meth:`~WflowSedimentModel.setup_lulcmaps`:
  Main method to setup LULC maps using lookup tables.
- :py:meth:`~WflowSbmModel.setup_lulcmaps_from_vector` and
  :py:meth:`~WflowSedimentModel.setup_lulcmaps_from_vector`: Similar to the above but starts
  with rasterizing the LULC vector data to a user-defined resolution.
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

.. code-block:: csv

    esa,description,landuse,vegetation_kext,land_manning_n,soil_compacted_fraction,vegetation_root_depth,vegetation_leaf_storage,vegetation_wood_storage,land_water_fraction,vegetation_crop_factor,vegetation_feddes_alpha_h1,vegetation_feddes_h1,vegetation_feddes_h2,vegetation_feddes_h3_high,vegetation_feddes_h3_low,vegetation_feddes_h4,erosion_usle_c,Cov_River
    10,Tree cover,10,0.8,0.5,0,406,0.23,0.09,0,1.1,1,0,-100,-400,-1000,-16000,0.0012
    20,Shrubland,20,0.7,0.5,0,410,0.1,0.05,0,1.05,1,0,-100,-400,-1000,-16000,0.06
    30,Grassland,30,0.6,0.2,0,106.8,0.1,0.01,0,1,1,0,-100,-400,-1000,-16000,0.04
    40,Cropland,40,0.6,0.15,0,390.4,0.077,0.005,0,1.15,0,0,-100,-400,-1000,-16000,0.3
    50,Built-up,50,0.6,0.015,0.9,257.4,0.1,0.03,0,1,1,0,-100,-400,-1000,-16000,0.001
    60,Bare / sparse vegetation,60,0.6,0.015,0,10.7,0.1,0.03,0,0.5,1,0,-100,-400,-1000,-16000,0.35
    70,Snow and Ice,70,0,0.01,0,0,0,0,0,1,1,0,-100,-400,-1000,-16000,0
    80,Permanent water bodies,80,0,0.01,0,0,0,0,1,1.05,1,0,-100,-400,-1000,-16000,0
    90,Herbaceous wetland,90,0.6,0.125,0,106.8,0.1,0.01,0,1.2,1,0,-100,-400,-1000,-16000,0.001
    95,Mangroves,95,0.8,0.5,0,369,0.23,0.09,0.5,1.05,1,0,-100,-400,-1000,-16000,0.008
    100,Moss and lichen,100,0.6,0.085,0,136.9,0.09,0,0,1.05,1,0,-100,-400,-1000,-16000,0.001
    0,No data,0,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999

Example usage
-------------
Here is an example of how to use the ``setup_lulcmaps`` method in a model setup workflow:





More examples can be found in the following notebooks:

- :ref:`Update land use <example-update_model_landuse>`
- :ref:`Add water demands and allocations (with paddy landuse) <example-update_model_water_demand>`

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
| leaf_storage | Specific leaf storage [mm]                                     | 0.02-0.2 | `Zhong et al. (2022) <https://doi.org/10.5194/hess-26-5647-2022>`_                |
+--------------+----------------------------------------------------------------+----------+-----------------------------------------------------------------------------------+
| wood_storage | Fraction of wood in the vegetation/plant [-]                   | 0.0-0.5  | `Zhong et al. (2022) <https://doi.org/10.5194/hess-26-5647-2022>`_                |
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

Leaf and wood storage
*********************

Previous values were derived from `Pitman (1989) <https://doi.org/10.5194/hess-15-3355-2011>`_
and `Liu (1998) <https://doi.org/10.1016/S0022-1694(98)00115-2>`_ . Starting from version 1, the
default lookup tables use updated values based on a literature review by
`Zhong et al. (2022) <https://doi.org/10.5194/hess-26-5647-2022>`_ (supplement values with more
details are available).

Note that for land use types with mixed (e.g urban) or sparse vegetation, the actual values will be scaled
with LAI.

.. list-table::
   :header-rows: 1

   * - Vegetation / Crop type
     - Leaf storage [mm]
     - Wood storage [-]
   * - Needleleaf forest
     - 0.29
     - 0.09
   * - Evergreen broadleaf forest
     - 0.20
     - 0.09
   * - Deciduous broadleaf forest
     - 0.18
     - 0.09
   * - Mixed forest
     - 0.20
     - 0.09
   * - All forest
     - 0.23
     - 0.09
   * - Short vegetation (crops, grass, shrub)
     - 0.10
     - 0.03 (0.01 - 0.05)
   * - Maize
     - 0.077
     - 0.005
   * - Rice
     - 0.042
     - 0.005

Evapotranspiration parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameters related to vegetation evaporation and transpiration.

+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| Parameter       | Description                                         | Range                           | Reference                                                                                     |
+=================+=====================================================+=================================+===============================================================================================+
| crop_factor     | Crop coefficient [-]                                | 0.3 - 1.25                      | `Allen et al. (1998) <https://www.fao.org/4/x0490e/x0490e0b.htm>`_                            |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| root_depth      | Length of vegetation roots [mm]                     | 100 - 5000                      | `Fan et al. (2016) <https://www.mdpi.com/2077-0472/14/4/532>`_                                |
|                 |                                                     |                                 | `Schenk and Jackson (2002) <https://doi.org/10.1890/0012-9615(2002)072[0311:TGBOR]2.0.CO;2>`_ |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_alpha_h1 | Root water uptake reduction at pressure head h1 [-] | 0 (crop) - 1 (other)            | `van Dam et al. (1997) <https://edepot.wur.nl/222782>`_                                       |
|                 |                                                     |                                 | `Singh et al. (2003) <https://www.academia.edu/download/102602419/19325.pdf>`_                |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h1       | Critical pressure head h1 - anorexic condition [cm] | 100 (paddy) - 0 (other)         | `van Dam et al. (1997) <https://edepot.wur.nl/222782>`_                                       |
|                 |                                                     |                                 | `Singh et al. (2003) <https://www.academia.edu/download/102602419/19325.pdf>`_                |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h2       | Critical pressure head h2 - field capacity [cm]     | 55 (paddy) - -100 (other)       | `van Dam et al. (1997) <https://edepot.wur.nl/222782>`_                                       |
|                 |                                                     |                                 | `Singh et al. (2003) <https://www.academia.edu/download/102602419/19325.pdf>`_                |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h3_high  | Critical pressure head h3 (high) [cm]               | -160 (paddy) - -400 (other)     | `van Dam et al. (1997) <https://edepot.wur.nl/222782>`_                                       |
|                 |                                                     |                                 | `Singh et al. (2003) <https://www.academia.edu/download/102602419/19325.pdf>`_                |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h3_low   | Critical pressure head h3 (low) [cm]                | -250 (paddy) - -1000 (other)    | `van Dam et al. (1997) <https://edepot.wur.nl/222782>`_                                       |
|                 |                                                     |                                 | `Singh et al. (2003) <https://www.academia.edu/download/102602419/19325.pdf>`_                |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+
| feddes_h4       | Critical pressure head h4 - wilting point [cm]      | -15000 (paddy) - -16000 (other) | `van Dam et al. (1997) <https://edepot.wur.nl/222782>`_                                       |
|                 |                                                     |                                 | `Singh et al. (2003) <https://www.academia.edu/download/102602419/19325.pdf>`_                |
+-----------------+-----------------------------------------------------+---------------------------------+-----------------------------------------------------------------------------------------------+

**Crop factor**

The factor or FAO-56 crop coefficient kc is used to scale reference evapotranspiration (ET0)
to crop evapotranspiration (ETc) as follows: ETc = Kc * ET0. In Wflow, kc is used as a maximum
value valid for a full cover of a vegetation/crop type (i.e. kc is not dependant on crop growth stage
or soil cover). Within Wflow, kc will be scaled further based on the actual vegetation cover fraction
(from LAI) to get the actual crop coefficient used for ETc calculation.

Detailed values of kc can be found for different crop types in the
`FAO guidelines <https://www.fao.org/4/x0490e/x0490e0b.htm>`. As most LULC
maps do not distinguish between crop types, an average value representing the most common crops
in your study area should be used. In the default lookup tables, 1.15 is used for cropland areas
(based on an average value for cereals and oil crops), and 1.2 for paddy/rice fields.


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

Critical pressure heads for rice are taken after Singh et al. (2003). For other vegetation,
the default values from Wflow.jl are used. These are now vegetation independent and are taken
as the default complete saturation (h1=0 cm), field capacity (h2=-100 cm) and wilting point
(h4=-16000 cm). The h3 values are set to -400 cm (high) and -1000 cm (low) but these are largely
dependent on the type of vegetation.

Examples can be found in annexes C and D of Van Dam et al. (1997). Here are examples for
the most common crops [cm]:

.. list-table::
   :header-rows: 1

   * - Crop
     - h1
     - h2
     - h3_high
     - h3_low
     - h4
   * - Potatoes
     - -10
     - -25
     - -320
     - -600
     - -16000
   * - Sugar beet
     - -10
     - -25
     - -320
     - -600
     - -16000
   * - Wheat
     - 0
     - -1
     - -500
     - -900
     - -16000
   * - Pasture
     - -10
     - -25
     - -200
     - -800
     - -8000
   * - Corn
     - -15
     - -30
     - -325
     - -600
     - -8000

Manning Roughness
^^^^^^^^^^^^^^^^^
Manning's N values are used to represent roughness of the land surface for overland flow.
Estimations per landuse class can be found in literature such as:

+-----------+-----------------------------+-------------+--------------------------------------------------------------------------+
| Parameter | Description                 | Range       | Reference                                                                |
+===========+=============================+=============+==========================================================================+
| manning_n | Manning Roughness [m-1/3 s] | 0.008-0.96  | `Engman (1986) <https://doi.org/10.1061/(ASCE)0733-9437(1986)112:1(39)>` |
|           |                             |             | `Kilgore (1997) <http://hdl.handle.net/10919/35777>`_                    |
|           |                             |             | `Cronshey (1986) <https://www.nrc.gov/docs/ML1421/ML14219A437.pdf>`_                                                          |
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
     - 0.01 (smooth bare soil or bare sand) / 0.011 (concrete) - 0.020 (gravel)
   * - Fallow (no residue)
     - 0.05
     - 0.05
     - 0.05
   * - Cropland
     - 0.06 - 0.17 (depending on residue cover)
     - 0.032 (wheat) / 0.08 (corn) - 0.2 (depending on tillage)
     - 0.1 - 0.4 (small grain) / 0.07 - 0.2 (row crops)
   * - Grassland
     - 0.15 (short) - 0.24 (dense)
     - 0.046 (grass) / 0.1 (pasture)
     - 0.15 (short) - 0.24 (dense)
   * - Forest
     - 0.4 - 0.8 (depending on underbrush)
     - 0.6
     -
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
For soil erosion, the soil cover-management factor USLE C can be estimated for different
land use / vegetation type.

+----------------+----------------------------------+-------------+-----------------------------------------------------------------------------+
| Parameter      | Description                      | Range       | Reference                                                                   |
+================+==================================+=============+=============================================================================+
| erosion_usle_c | USLE cover management factor [-] | 0.001 - 1.0 | `Panagos et al. (2015) <https://doi.org/10.1016/j.landusepol.2015.05.021>`_ |
|                |                                  |             | `Bosco et al. (2015) <https://doi.org/10.5194/nhess-15-225-2015>`_          |
|                |                                  |             | `Gericke et al. (2015) <https://doi.org/10.1080/15715124.2014.1003302>`_    |
+----------------+----------------------------------+-------------+-----------------------------------------------------------------------------+

Examples of USLE C values for different land use types different sources:

.. list-table::
   :header-rows: 1

   * - Land use
     - Panagos
     - Bosco
     - Gericke
   * - Wheat
     - 0.20
     -
     -
   * - Maize
     - 0.38
     -
     -
   * - Rice
     - 0.15
     - 0.15
     - 0.05
   * - Potatoes or sugar beet
     - 0.34
     -
     -
   * - Oilseeds
     - 0.28
     -
     -
   * - All crops
     - 0.233 (0.2 - 0.5)
     - 0.2 (irrigated) / 0.335 (rainfed)
     - 0.18 - 0.24 (irrigated) / 0.3 - 0.4 (rainfed)
   * - Vineyards
     - 0.3527 (0.15 - 0.45)
     - 0.45
     - 0.5
   * - Fruit trees and berries
     - 0.2188 (0.1 - 0.3)
     - 0.35
     - 0.4
   * - Olive groves
     - 0.2273 (0.1 - 0.3)
     - 0.35
     - 0.4
   * - Agro-forestry areas
     - 0.0881 (0.03 - 0.13)
     - 0.2
     - 0.23 - 0.3
   * - Broad-leaved forest
     - 0.0013 (0.0001 - 0.003)
     - 0.0025
     - 0.005 - 0.008
   * - Coniferous forest
     - 0.0011 (0.0001 - 0.003)
     - 0.0015
     - 0.005 - 0.008
   * - Mixed forest
     - 0.0011 (0.0001 - 0.003)
     - 0.002
     - 0.005 - 0.008
   * - Pastures
     - 0.0903 (0.05 - 0.15)
     - 0.01
     - 0.01 - 0.005
   * Natural grasslands
     - 0.0435 (0.01 - 0.08)
     - 0.005
     - 0.01 - 0.05
   * - Moors and heathland
     - 0.0420 (0.01 - 0.1)
     - 0.05
     - 0.01 - 0.05
   * - Shrubland
     - 0.0623 (0.01 - 0.1)
     - 0.04
     - 0.01 - 0.05
   * - Bare rocks
     - 0
     -
     - 0
   * - Sparse vegetation
     - 0.2652 (0.1 - 0.45)
     - 0.3
     - 0.35
   * - Burnt areas
     - 0.3427 (0.1 - 0.55)
     - 0.3
     - 0.35
   * - Glaciers and perpetual snow
     - 0
     - 0.001
     - 0

References
----------

- Allen RG, Pereira LS, Raes D, Smith M (1998) Crop evapotranspiration guidelines for computing
  crop water requirements. FAO Irrig Drain Pap 56. FAO, Rome, p 300
- Bosco, C., de Rigo, D., Dewitte, O., Poesen, J., and Panagos, P. (2015). Modelling soil
  erosion at European scale: towards harmonization and reproducibility, Nat. Hazards Earth Syst.
  Sci., 15, 225–245, https://doi.org/10.5194/nhess-15-225-2015
- Corbari, C., Ravazzani, G., Galvagno, M., Cremonese, E., & Mancini, M. (2017). Assessing
  crop coefficients for natural vegetated areas using satellite data and eddy covariance
  stations. Sensors, 17(11), 2664.
- Cronshey, R. (1986). Urban hydrology for small watersheds (No. 55). US Department of Agriculture,
  Soil Conservation Service, Engineering Division.
- van Dam, J.C., Huygen, J., Wesseling, J.G., Feddes, R.A., Kabat, P., van Walsum, P.E.V., Groenendijk, P.,
  and van Diepen, C.A., 1997. Theory of SWAP version 2.0: Simulation of water flow, solute transport
  and plant growth in the soil-water-atmosphere-plant environment. Wageningen Agricultural
  University, The Netherlands, Report 71.
- van Dijk, A. I. J. M., & Bruijnzeel, L. A. (2001). Modelling rainfall interception by vegetation
  of variable density using an adapted analytical model. Part 2. Model validation for a tropical
  upland mixed cropping system. Journal of Hydrology, 247(3-4), 239–262.
- Engman, E. (1986). Roughness coefficients for routing surface runoff. Journal of Irrigation
  and Drainage Engineering, 112(1), 39-53. https://doi.org/10.1061/(ASCE)0733-9437(1986)112:1(39)
- Fan, J., McConkey, B., Wang, H., & Janzen, H. (2016). Root distribution by depth for temperate
  agricultural crops. Field Crops Research, 189, 68–74. https://doi.org/10.1016/j.fcr.2016.02.013
- Feddes, R.A., Kowalik, P.J. and Zaradny, H., 1978, Simulation of field water use and crop yield,
  Pudoc, Wageningen, Simulation Monographs.
- Gericke, A. (2015). Soil loss estimation and empirical relationships for sediment delivery
  ratios of European river catchments. International Journal of River Basin Management,
  13(2), 179–202. https://doi.org/10.1080/15715124.2014.1003302
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
- Panagos, P., Borrelli, P., Meusburger, K., Alewell, C., Lugato, E., & Montanarella, L. (2015).
  Estimating the soil erosion cover-management factor at the European scale. Land Use Policy,
  48, 38–50. https://doi.org/10.1016/j.landusepol.2015.05.021
- Pereira, L.S., Paredes, P. & Espírito-Santo, D. (2024a). Crop coefficients of natural wetlands
  and riparian vegetation to compute ecosystem evapotranspiration and the water balance.
  Irrig Sci 42, 1171–1197. https://doi.org/10.1007/s00271-024-00923-9
- Pereira, L.S., Paredes, P., Espírito-Santo, D. et al. (2024b). Actual and standard crop
  coefficients for semi-natural and planted grasslands and grasses: a review aimed at
  supporting water management to improve production and ecosystem services. Irrig Sci 42,
  1139–1170. https://doi.org/10.1007/s00271-023-00867-6
- Pereira, L.S., Paredes, P., Oliveira, C.M. et al. (2024c). Single and basal crop coefficients
  for estimation of water use of tree and vine woody crops with consideration of fraction of
  ground cover, height, and training system for Mediterranean and warm temperate fruit and
  leaf crops. Irrig Sci 42, 1019–1058. https://doi.org/10.1007/s00271-023-00901-7
- Pitman, J. (1989). Rainfall interception by bracken in open habitats—Relations between
  leaf area, canopy storage and drainage rate. Journal of Hydrology, 105(3-4), 317–334.
  https://doi.org/10.1016/0022-1694(89)90111-X
- Schenk, H. J., & Jackson, R. B. (2002). The global biogeography of roots. Ecological
  Monographs, 72(3), 311–328. https://doi.org/10.1890/0012-9615(2002)072[0311:TGBOR]2.0.CO;2
- Singh, R., Van Dam, J. C., & Jhorar, R. K. (2003). Water and salt balances at farmer fields.
  Water productivity of irrigated crops in Sirsa district, India. Integration of remote sensing,
  crop and soil models and geographical information systems.
- Zhong, F., Jiang, S., van Dijk, A. I. J. M., Ren, L., Schellekens, J., and Miralles, D. G.
  (2022). Revisiting large-scale interception patterns constrained by a synthesis of global
  experimental data, Hydrol. Earth Syst. Sci., 26, 5647–5667. https://doi.org/10.5194/hess-26-5647-2022
