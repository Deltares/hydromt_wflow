{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

Summary of Methods and Attributes
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component class
     - Methods
     - Attributes

   * - :class:`~{{ fullname }}`
     - {% for m in methods %}:meth:`~{{ fullname }}.{{ m }}`{% if not loop.last %}, {% endif %}{% endfor %}
     - {% for a in attributes %}:attr:`~{{ fullname }}.{{ a }}`{% if not loop.last %}, {% endif %}{% endfor %}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource
