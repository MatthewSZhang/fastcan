"""Cython extension
Use in the current directory:
python setup.py build_ext --inplace
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

DISTNAME = "fastcan"


cython_exts = [
    Extension(
        name=DISTNAME + "._cancorr_fast",
        sources=[DISTNAME + "/_cancorr_fast.pyx"],
    ),
]


if __name__ == "__main__":
    setup(
        packages=find_packages(where=DISTNAME),
        ext_modules=cythonize(cython_exts),
        include_dirs=[np.get_include()],
    )
