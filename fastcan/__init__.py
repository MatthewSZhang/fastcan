"""
The :mod:`fastcan` module implements algorithms, including
"""

from ._fastcan import FastCan
from ._utils import ols, ssc

__all__ = [
    "FastCan",
    "ssc",
    "ols",
]
