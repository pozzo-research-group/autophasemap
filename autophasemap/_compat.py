"""Compatibility helpers for NumPy/SciPy API changes.

- ``numpy.trapz`` was deprecated in NumPy 2.0 and removed in NumPy 2.4;
  ``numpy.trapezoid`` replaces it (it does not exist before NumPy 2.0).
- ``scipy.integrate.cumtrapz`` was removed in SciPy 1.14;
  ``scipy.integrate.cumulative_trapezoid`` (SciPy >= 1.6) replaces it.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

try:
    trapezoid = np.trapezoid
except AttributeError:  # NumPy < 2.0
    trapezoid = np.trapz

__all__ = ["trapezoid", "cumulative_trapezoid"]
