Design Principles and Guidelines
================================

In this file, we provide the design principles and guidelines for the ``hydromt_wflow`` package.
The package is designed to be **modular**, **extensible**, and **maintainable**, following best practices in software design.

.. _software_design:

Overview
--------

HydroMT-Wflow is used as a Python library or as a command-line tool using YAML (.yml) configuration files.

To keep usage relatively simple, the package is designed around a single ``WflowModel`` object, which is used to build, modify, and export Wflow models.
Users interact with the package primarily through YAML files, which are translated into function calls on the ``WflowModel`` class.

This means that end users do not need to worry or know about the underlying components, workflows, or processes.
The only way that people can interact with / use Hydromt-Wflow, will be the public methods in ``WflowModel``.
Even users that prefer the yml files will be using only the public functions (or ``hydromt_step``) of ``WflowModel``, which are a 1 to 1 mapping to these functions as well.

WflowModel as Orchestrator
--------------------------

The ``WflowModel`` class is designed to serve as an orchestrator or manager for the other parts of the codebase.

It combines components, workflows, and processes, but does **not** contain the actual implementation logic or complexity.

All the methods in ``WflowModel`` are simple and delegate the real work to the respective components.

Currently, some ``WflowModel`` methods do combine logic from multiple components.

For instance, the ``forcing`` component may read configuration settings from the ``config`` component, then use the ``grid`` component to process forcing data.

This design choice is intentional: **any logic that requires coordination between multiple components should be placed in ``WflowModel``.**

Typical Method Structure in WflowModel
--------------------------------------

Every method in the ``WflowModel`` class should generally follow these steps:

1. **Validation**

    Validate the state of the model, but also the method inputs to ensure they are correct and complete.
    This includes checking types, formats, values, and asserting read/write modes.

2. **Data Retrieval**

    Retrieve any necessary data from the ``DataCatalog`` or other components.
    For example, reading data from the catalog or retrieving internal component state.

3. **Delegation**

    Based on the inputs and retrieved data, determine which components, methods, or workflows to call, then pass the validated inputs to the appropriate methods.

    This may involve calling methods from ``config``, ``grid``, ``forcing``, ``geoms``, or any workflow logic.

    **Important**: Each component should only use its own methods or public methods from ``WflowModel``, components must **not** access or depend on other components directly.

4. **Output Handling**

    (Post-)process and return/store the outputs from the invoked operations.
    Outputs may be stored in components or returned to the user.

An example method structure using made up method names in ``WflowModel`` might look like this:

.. code-block:: python

   import workflows
   from components import ForcingComponent, ConfigComponent, DataCatalog, GridComponent

   class WflowModel:
      forcing: ForcingComponent
      config: ConfigComponent
      data_catalog: DataCatalog
      grid: GridComponent

      ...

      def example_method(self, input_data: str):
         # Step 1: Validation
         self._assert_read_mode()

         # Step 2: Data retrieval
         config_data = self.config.get(input_data)
         data = self.data_catalog.get_data(config_data)

         # Step 3: Delegation to components
         partial_result = self.grid.process_data(data)
         result = workflows.example_workflow(partial_result)

         # Step 4: Output handling
         self.forcing.store_result(result)

The above structure ensures that each method is clear, focused, and follows a consistent pattern.
It also allows the components and workflows to focus on their specific tasks without worrying about the overall orchestration.

Component Design Principles
---------------------------

Each component in the system should follow these principles:

- **Encapsulation and Independence**

  Each component is self-contained and independent.
  It must not call or depend on any other components or internal attributes of ``WflowModel``.
  This ensures components are easily replaceable and extensible without impacting the system as a whole.
  For example:

  - ``grid`` must not access ``config``

  - ``forcing`` must not use ``grid``

  Instead, components should expose methods that can be called by ``WflowModel`` that will take in and or return the necessary data, which can then be passed to other components or workflows as needed.

- **Strict Typing and Interfaces**

  Component methods should have narrow and well-defined type signatures.
  This improves clarity, maintainability, and testability.
  Broad or ambiguous argument types (e.g., ``data_like`` in the datacatalog, which might be a ``str``, ``Path``, ``GeoDataFrame``, ``xr.Dataset``, ``np.ndarray``, or ``None``) are **not allowed** in component methods.

  Any such type resolution or transformation must be handled in the ``WflowModel`` before calling component methods.
  This also means that components might have multiple methods for different data types.
