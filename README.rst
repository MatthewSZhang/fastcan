FastCan: A Fast Canonical-Correlation-Based Feature Selection Method
====================================================================

FastCan is a python implementation of the paper

Installation and Uninstallation
-------------------------------

.. note::

    For Windows user, the **Build Tools for Visual Studio** is required before the installation of FastCan.

Install **FastCan** in editable mode:

#. Open ``Command Prompt`` or ``Anaconda Prompt``
#. cd to ``\python-package``, where ``setup.py`` is located
#. Run ``pip install -e .`` (Note: there is a DOT at the end)

Uninstall **FastCan**:

#. Open ``Command Prompt`` or ``Anaconda Prompt``
#. Run ``pip uninstall fastcan``

Add extraPahts to enable Pylance extension in VS Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Pylance extension will fail to recognize the package install with ``pip install -e .``.
To sovle this problem, add the path where the parent directory of ``fastcan`` to ``.vscode/settings.json``, e.g.,::

    "python.analysis.extraPaths": ["WORKING_DIRECTORY"]

You can also use strict editable mode of pip install to solve this problem, i.e.,

.. code-block:: shell

    pip install -e . --config-settings editable_mode=strict

Makefile
--------

.. note::

    Some commands, e.g., ``rm``, are only avaible in ``bash``.


Glossary
--------
.. glossary::

    fit
        The ``fit`` **method** is align with scikit-learn.

    random_state
        The ``random_state`` **parameter** is align with scikit-learn.

Reference
---------
#. Sphinx hyperlinks: |sphinx_link|_
