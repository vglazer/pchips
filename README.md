# PCHIPs

This package provides a Python implementation of the Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) algorithm.

This code is a direct port of the MATLAB implementation based on the paper **"Accurate Monotone Cubic Interpolation"** by Hung T. Huynh (NASA Technical Memorandum 103789).

The original MATLAB code was written by **Victor Glazer** under the supervision of **Christina C. Christara** as part of an NSERC USRA project at the University of Toronto.

## Features

- Monotonicity-preserving piecewise cubic Hermite interpolation.
- Selectable derivative approximation methods ('cubic' or 'quartic').
- Selectable monotonicity constraints ('M3' or 'M4').
