Design Principles and Guidelines
================================

In this file, we provide the design principles and guidelines for the ``hydromt_wflow`` package.
The package is designed to be **modular**, **extensible**, and **maintainable**, following best practices in software design.

.. _software_design:

Overview
--------

HydroMT-Wflow is used as a Python library or as a command-line tool using YAML (.yml) configuration files.

To keep usage relatively simple, the package is designed around a single ``WflowBaseModel`` object, which should never be used directly by end-users.
Its subclasses, which are ``WflowSbmModel`` and ``WflowSedimentModel``, are used to build, modify, and export Wflow models.

To allow for extensions and customizations for specific use cases, the ``WflowBaseModel`` class contains the generic components, methods and properties that are common across all Wflow models.
Any specific implementations and behaviors are meant to be defined in subclasses.
From now, we will refer to these models collectively as ``Model`` for simplicity, but it is important to note that the principles and guidelines apply to all these classes.

Users interact with the package primarily through YAML files, which are translated into function calls on the ``Model`` class.
This means that end users do not need to worry or know about the underlying components, workflows, or processes.
The only way that people should interact with / use Hydromt-Wflow, will be the public methods in ``Model``.
Functions used in yml files correspond 1-to-1 to public model functions decorated with ``hydromt_step```.
Only the public methods of ``Model`` are allowed to be a ``hydromt_step``.
Components and workflows cannot be used directly as ``hydromt_step`` callables.

Model as Orchestrator
------------------------------

The ``Model`` class is designed to serve as an orchestrator or manager for the other parts of the codebase.

It combines components, workflows, and processes, but does **not** contain the actual implementation logic or complexity.

All the methods in ``Model`` are simple and delegate the real work to the respective components and workflows.

Currently, some ``Model`` methods do combine logic from multiple components.

For instance, the forcing component may need configuration settings from the config component, and then need to use the data from the grid component. However, to keep these components ignorant of each other, the WflowBaseModel class will be the connection.

This design choice is intentional: **any interaction between multiple components has to go through public methods on the ``Model`` class.**
Methods often require knowledge of multiple components and their interactions (e.g. write_forcing requires access to the config and grid).
To encourage reduced coupling, components are allowed to call public functions on the Model class.
This way ``set_config``, ``get_config``, ``set_grid``, etc, can be used without needing to know about the internals of other components.

Typical Method Structure in Model
------------------------------------------

Every method in the ``Model`` class should generally follow these steps:

1. **Data Retrieval**

    Retrieve any necessary data from the ``DataCatalog`` or other components.
    For example, reading data from the catalog or retrieving internal component state.

2. **Delegation**

    Based on the inputs and retrieved data, determine which components, methods, or workflows to call, then pass the validated inputs to the appropriate methods.

    This may involve calling methods from ``config``, ``grid``, ``forcing``, ``geoms``, or any workflow logic.

    **Important**: Each component should only use its own methods or public methods from ``Model``, components must **not** access or depend on other components directly.

3. **Output Handling**

    (Post-)process and return/store the outputs from the invoked operations.
    Outputs may be stored in components or returned to the user.

An example method structure using made up method names in ``Model`` might look like this:

.. code-block:: python

  import workflows
  from wflow_base import WflowBaseModel
  from components import ForcingComponent, ConfigComponent, DataCatalog, GridComponent

  class WflowExampleModel(WflowBaseModel):
    forcing: ForcingComponent
    config: ConfigComponent
    data_catalog: DataCatalog
    grid: GridComponent
    ...
    def example_setup_method(self, input_data: str, model_option1: bool = true):
      # Step 1: Data retrieval
      config_data = self.config.get("setting1") # eg get starttime and endtime
      data = self.data_catalog.get_data(input_data, config_data)

      # Step 2: Delegation to components and workflows
      partial_result = workflows.example_workflow1(data)
      result = workflows.example_workflow2(partial_result, self.grid.data, model_option1)

      # Step 3: Output handling
      self.config.set(model.option1, model_option1)
      if "var1" in result:
        self.config.set(model.dovar1, true)
      rename_dict = {k: v for k, v in self._MAPS if k in result} # from hydromt to wflow name
      self.forcing.set(result.rename(rename_dict))

The above structure ensures that each method is clear, focused, and follows a consistent pattern.
It also allows the components and workflows to focus on their specific tasks without worrying about the overall orchestration.

Component Design Principles
---------------------------

Each component in the system should follow these principles:

- **Encapsulation and Independence**

  Each component is self-contained and independent.
  It must not call or depend on any other components or internal attributes of ``Model``.
  This ensures components are easily replaceable and extensible without impacting the system as a whole.
  For example:

  - ``grid`` must not access ``config``

  - ``forcing`` must not use ``grid``

  Instead, components should expose methods that can be called by ``Model`` that will take in and or return the necessary data, which can then be passed to other components or workflows as needed.

- **Strict Typing and Interfaces**

  Component methods should have narrow and well-defined type signatures.
  This improves clarity, maintainability, and testability.
  Broad or ambiguous argument types (e.g., ``data_like`` in the datacatalog, which might be a ``str``, ``Path``, ``GeoDataFrame``, ``xr.Dataset``, ``np.ndarray``, or ``None``) are **not allowed** in component methods.

  Any such type resolution or transformation must be handled in the ``Model`` before calling component methods.
  This also means that components might have multiple methods for different data types.

- **Validation**
  Validate the state of the component & model (read/write mode), but also the method inputs to ensure they are correct and complete.
  This can include checking types, formats, values, and asserting read/write modes.


Workflows
---------
Workflows are functions that combine primitive data and model operations into higher-level processes.
They are defined in the ``workflows`` module and can be called from ``Model`` methods.
Workflows should follow these principles:
- **Single Responsibility**: Each workflow should perform a specific task or process.
- **Reusability**: Workflows should be designed to be reusable across different components and methods.
- **No Direct Component Access**: Workflows should not directly access or modify component states. Instead, they should operate on data passed to them from ``WflowBaseModel``.
- **Validation**: Workflows should validate their inputs and outputs to ensure correctness.
- **Naming Conventions**: workflows work with the hydromt-naming conventions, and should also handle the renaming between hydromt-names and wflow names.
