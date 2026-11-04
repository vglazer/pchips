import numpy as np
import scipy.interpolate

class PchipInterpolator(scipy.interpolate.CubicHermiteSpline):
    """
    PchipInterpolator: H.T. Huynh's accurate monotone cubic interpolant.

    This class is a drop-in replacement for `scipy.interpolate.PchipInterpolator`
    but uses H.T. Huynh's algorithm instead of F. N. Fritsch and J. Butland's.
    Like SciPy's version, it is a `CubicHermiteSpline`, so evaluation, `derivative`,
    `antiderivative`, `integrate`, `roots`, `solve` and `extrapolate` all behave the same.

    The interpolant is uniformly third or fourth-order accurate, and it preserves
    monotonicity: with the default M3 constraint, monotone data give a monotone
    interpolant. Unlike SciPy's version, however, it may overshoot the data near
    strict local extrema, so it is not bounded by the data. For example, it can
    dip below zero next to a small positive minimum.

    With fewer than five data points, the derivatives are those of
    `scipy.interpolate.PchipInterpolator`.
    """

    def __init__(self, x, y, axis=0, extrapolate=None, *, approx_order='cubic', mono_constraint='M3'):
        """
        Initializes the interpolator.

        Args:
            x (array_like): The knot vector, strictly increasing.
            y (array_like): The real data. Its length along `axis` must match `x`.
            axis (int): The axis of `y` along which the data vary (default 0).
            extrapolate (bool or 'periodic'): Whether to extrapolate outside the
                                              data range (default True).
            approx_order (str): The order of the derivative approximation formula.
                                Supported: 'cubic' (default), 'quartic'.
            mono_constraint (str): The monotonicity constraint type.
                                   Supported: 'M3' (default), 'M4'.
        """
        if approx_order not in ('cubic', 'quartic'):
            raise ValueError(f"Unsupported approx_order: {approx_order}")
        if mono_constraint not in ('M3', 'M4'):
            raise ValueError(f"Unsupported mono_constraint: {mono_constraint}")
        self.approx_order = approx_order
        self.mono_constraint = mono_constraint

        # Validate the input the same way as scipy.interpolate.PchipInterpolator
        x, y = map(np.asarray, (x, y))
        if np.issubdtype(x.dtype, np.complexfloating):
            raise ValueError("`x` must contain real values.")
        if np.issubdtype(y.dtype, np.complexfloating):
            raise ValueError("`PchipInterpolator` only works with real values for `y`.")
        x = x.astype(float)
        y = y.astype(float)
        axis = axis % y.ndim
        if x.ndim != 1:
            raise ValueError("`x` must be 1-dimensional.")
        if x.shape[0] < 2:
            raise ValueError("`x` must contain at least 2 elements.")
        if x.shape[0] != y.shape[axis]:
            raise ValueError(f"The length of `y` along `axis`={axis} doesn't match the length of `x`")
        if not np.all(np.isfinite(x)):
            raise ValueError("`x` must contain only finite values.")
        if not np.all(np.isfinite(y)):
            raise ValueError("`y` must contain only finite values.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("`x` must be strictly increasing sequence.")

        self.x = x
        self.n = len(x)
        y = np.moveaxis(y, axis, 0)

        if self.n < 5:
            # Too few points for Huynh's derivative approximations
            dydx = scipy.interpolate.PchipInterpolator(x, y).derivative()(x)
        else:
            # Compute the derivatives for all columns at once
            columns = y.reshape(self.n, int(np.prod(y.shape[1:])))
            d = self._approxder(columns)
            dydx = self._perturbder(d, columns).reshape(y.shape)

        super().__init__(x, y, dydx, axis=0, extrapolate=extrapolate)
        self.axis = axis

    # The helpers below work on data with shape (n, number of columns).
    # Indexing with an array of k evaluates a formula at all of those k at once.

    def _vecdiffs(self, y):
        x = self.x
        s = np.diff(y, axis=0) / (x[1:] - x[:-1])[:, None]
        ds = np.diff(s, axis=0) / (x[2:] - x[:-2])[:, None]
        e = np.diff(ds, axis=0) / (x[3:] - x[:-3])[:, None]
        f = np.diff(e, axis=0) / (x[4:] - x[:-4])[:, None]
        return s, ds, e, f

    def _approxder(self, y):
        d = np.zeros_like(y)
        s, ds, e, f = self._vecdiffs(y)
        x = self.x[:, None]
        n = self.n

        if self.approx_order == 'cubic':
            k = np.arange(n - 3)
            d[k] = s[k] + ds[k]*(x[k] - x[k+1]) + \
                   e[k]*(x[k]**2 + x[k+1]*x[k+2] - x[k]*x[k+1] - x[k]*x[k+2])
            w1 = 3*e[n-4]
            w2 = 2*(ds[n-4] - e[n-4]*(x[n-4] + x[n-3] + x[n-2]))
            rest = s[n-4] - ds[n-4]*(x[n-4] + x[n-3]) + \
                   e[n-4]*(x[n-4]*x[n-3] + x[n-4]*x[n-2] + x[n-3]*x[n-2])
            k = np.arange(n - 3, n)
            d[k] = w1*x[k]**2 + w2*x[k] + rest
        elif self.approx_order == 'quartic':
            k = np.arange(n - 4)
            d[k] = s[k] + ds[k]*(x[k] - x[k+1]) + \
                   e[k]*(x[k]*(x[k] - x[k+1] - x[k+2]) + x[k+1]*x[k+2]) + \
                   f[k]*(x[k]**2*(x[k] - x[k+1] - x[k+2] - x[k+3]) + \
                         x[k]*(x[k+1]*x[k+2] + x[k+1]*x[k+3] + x[k+2]*x[k+3]) - \
                         x[k+1]*x[k+2]*x[k+3])
            w1 = 4*f[n-5]
            w2 = 3*(e[n-5] - f[n-5]*(x[n-5] + x[n-4] + x[n-3] + x[n-2]))
            w3 = 2*(ds[n-5] - e[n-5]*(x[n-5] + x[n-4] + x[n-3]) + \
                    f[n-5]*(x[n-5]*(x[n-4] + x[n-3] + x[n-2]) + \
                    x[n-4]*x[n-3] + x[n-4]*x[n-2] + x[n-3]*x[n-2]))
            rest = s[n-5] - ds[n-5]*(x[n-5] + x[n-4]) + \
                   e[n-5]*(x[n-5]*x[n-4] + x[n-5]*x[n-3] + x[n-4]*x[n-3]) - \
                   f[n-5]*(x[n-4]*x[n-3]*x[n-2] + \
                           x[n-5]*(x[n-4]*x[n-3] + x[n-4]*x[n-2] + x[n-3]*x[n-2]))
            k = np.arange(n - 4, n)
            d[k] = w1*x[k]**3 + w2*x[k]**2 + w3*x[k] + rest
        else:
            raise ValueError(f"Unsupported approx_order: {self.approx_order}")
        return d

    def _minmod(self, a, b):
        return (np.sign(a) + np.sign(b)) / 2 * np.minimum(np.abs(a), np.abs(b))

    def _ttilde(self, k, s, ds, emin, h):
        # Slopes at x[k] of the cubics on [x[k-1], x[k]] and [x[k], x[k+1]] (Huynh, 4.8a,b and 4.11)
        x = self.x[:, None]
        q1 = s[k-1] + h[k-1]*self._minmod(ds[k-2] + emin[k-3]*(x[k]   - x[k-2]),
                                    ds[k-1] + emin[k-2]*(x[k]   - x[k+1]))
        q2 = s[k]   - h[k]  *self._minmod(ds[k-1] + emin[k-2]*(x[k]   - x[k-1]),
                                    ds[k]   + emin[k-1]*(x[k]   - x[k+2]))
        return self._minmod(q1, q2)

    def _tmax(self, k, s, dmin, smin):
        # Bound which relaxes the monotonicity constraint near extrema (Huynh, 3.13 and 3.14)
        nder = 3
        x = self.x[:, None]
        p1 = s[k-1] + dmin[k-2]*(x[k] - x[k-1])
        p2 = s[k]   + dmin[k-1]*(x[k] - x[k+1])
        t = self._minmod(p1, p2)
        return t, np.sign(t) * np.maximum(nder * np.abs(smin[k-1]), (nder / 2) * np.abs(t))

    def _perturbder(self, d, y):
        nder = 3
        n = self.n
        dper = d.copy()
        s, ds, e, f = self._vecdiffs(y)
        h = np.diff(self.x)[:, None]
        smin = self._minmod(s[:-1], s[1:])
        dmin = self._minmod(ds[:-1], ds[1:])

        if self.mono_constraint not in ('M3', 'M4'):
            raise ValueError(f"Unsupported mono_constraint: {self.mono_constraint}")

        # Basic monotonicity constraint at the endpoints, MP constraint next to them (Huynh, section 5)
        dper[0] = self._minmod(d[0], nder * s[0])
        dper[n-1] = self._minmod(d[n-1], nder * s[n-2])
        dper[1] = self._minmod(d[1], nder * self._minmod(s[0], s[1]))
        dper[n-2] = self._minmod(d[n-2], nder*self._minmod(s[n-3], s[n-2]))

        if self.mono_constraint == 'M3':
            k = np.arange(2, n - 2)
            _, tmax = self._tmax(k, s, dmin, smin)
            dper[k] = self._minmod(d[k], tmax)
        else:
            # M3 constraint next to the MP points, M4 constraint in the interior
            k = np.array([2, n - 3])
            _, tmax = self._tmax(k, s, dmin, smin)
            dper[k] = self._minmod(d[k], tmax)
            emin = self._minmod(e[:-1], e[1:])
            k = np.arange(3, n - 3)
            t, _ = self._tmax(k, s, dmin, smin)
            tt = self._ttilde(k, s, ds, emin, h)
            zero = np.zeros_like(t)
            tmin = np.minimum(np.minimum(zero, nder*smin[k-1]), np.minimum((nder/2)*t, tt))
            tmax = np.maximum(np.maximum(zero, nder*smin[k-1]), np.maximum((nder/2)*t, tt))
            dper[k] = d[k] + self._minmod(tmin - d[k], tmax - d[k])
        return dper

pchip = PchipInterpolator


def pchip_interpolate(xi, yi, x, der=0, axis=0, *, approx_order='cubic', mono_constraint='M3'):
    """
    Convenience function for pchips interpolation, a drop-in replacement for
    `scipy.interpolate.pchip_interpolate`.

    Args:
        xi (array_like): The knot vector, strictly increasing.
        yi (array_like): The real data. Its length along `axis` must match `xi`.
        x (scalar or array_like): The points at which to evaluate the interpolant.
        der (int or list): The derivatives to evaluate (default 0, the values).
        axis (int): The axis of `yi` along which the data vary (default 0).
        approx_order (str): See `PchipInterpolator`.
        mono_constraint (str): See `PchipInterpolator`.
    """
    P = PchipInterpolator(xi, yi, axis=axis, approx_order=approx_order, mono_constraint=mono_constraint)
    if der == 0:
        return P(x)
    elif np.isscalar(der):
        return P.derivative(der)(x)
    else:
        return [P.derivative(nu)(x) for nu in der]
