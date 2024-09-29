"""
The :mod:`fastcan` module implements algorithms, including
"""

from ._fastcan import FastCan
from ._utils import ssc, ols

__all__ = [
    "FastCan",
    "ssc",
    "ols",
]
