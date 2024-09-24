"""
The :mod:`fastcan` module implements algorithms, including
"""

from ._fastcan import FastCan
from ._ssc import ssc

__all__ = [
    "FastCan",
    "ssc",
]
