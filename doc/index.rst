+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+
| .. figure:: images/sheffield-logo.svg   | .. figure:: images/tongji-logo.jpg      | .. figure:: images/DRG-logo.png         | .. figure:: images/shmc.png             |
|   :target: https://www.sheffield.ac.uk/ |   :target: https://www.tongji.edu.cn/   |   :target: https://drg.ac.uk/           |   :target: https://shmc.tongji.edu.cn/  |
|   :figwidth: 70pt                       |   :figwidth: 70pt                       |   :figwidth: 70pt                       |   :figwidth: 70pt                       |
|   :alt: Sheffield                       |   :alt: Tongji                          |   :alt: DRG                             |   :alt: SHMC                            |
|                                         |                                         |                                         |                                         |
|                                         |                                         |                                         |                                         |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+

.. include:: ../README.rst

Architecture Diagram
--------------------

.. uml:: diagram.puml
   :align: center

API Reference
-------------
.. autosummary::
   :toctree: generated/
   :template: module.rst

   fastcan
   fastcan.narx
   fastcan.utils

Useful Links
------------
.. toctree::
   :maxdepth: 2

   User Guide <user_guide>
   Examples <auto_examples/index>

API Compatibility
-----------------

The API of this library is align with scikit-learn.

.. |sklearn| image:: images/scikit-learn-logo-notext.png
   :width: 100pt
   :target: https://scikit-learn.org/

|sklearn|
