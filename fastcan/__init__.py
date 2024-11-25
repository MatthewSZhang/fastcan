"""
The :mod:`fastcan` module implements algorithms, including
"""

from ._extend import extend
from ._fastcan import FastCan
from ._narx import (
    Narx,
    make_narx,
    make_poly_features,
    make_poly_ids,
    make_time_shift_features,
    make_time_shift_ids,
    print_narx,
)
from ._refine import refine
from ._utils import ols, ssc

__all__ = [
    "FastCan",
    "ssc",
    "ols",
    "refine",
    "extend",
    "make_narx",
    "print_narx",
    "Narx",
    "make_poly_features",
    "make_poly_ids",
    "make_time_shift_features",
    "make_time_shift_ids",
]
