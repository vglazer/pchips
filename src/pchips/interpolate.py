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

        # Not self.x, which recent SciPy versions only allow to be set by CubicHermiteSpline.__init__
        self._knots = x
        self.n = len(x)
        y = np.moveaxis(y, axis, 0)

        if self.n < 5:
            # Too few points for Huynh's derivative approximations
            dydx = scipy.interpolate.PchipInterpolator(x, y).derivative()(x)
        else:
            # Compute the derivatives for all columns at once
            columns = y.reshape(self.n, int(np.prod(y.shape[1:])))
            diffs = self._vecdiffs(columns)
            d = self._approxder(diffs)
            dydx = self._perturbder(d, diffs).reshape(y.shape)

        super().__init__(x, y, dydx, axis=0, extrapolate=extrapolate)
        self.axis = axis

    # The helpers below work on data with shape (n, number of columns). Each formula is
    # evaluated for a range of nodes k at once: with at = _offsets(start, stop),
    # s[at(-1)] is s[k-1] for all k in range(start, stop), without copying.

    @staticmethod
    def _offsets(start, stop):
        return lambda offset: slice(start + offset, stop + offset)

    @staticmethod
    def _minmod(a, b):
        return np.where(np.signbit(a) == np.signbit(b), np.copysign(np.minimum(np.abs(a), np.abs(b)), a), 0.0)

    def _vecdiffs(self, y):
        x = self._knots
        s = np.diff(y, axis=0) / (x[1:] - x[:-1])[:, None]
        ds = np.diff(s, axis=0) / (x[2:] - x[:-2])[:, None]
        e = np.diff(ds, axis=0) / (x[3:] - x[:-3])[:, None]
        f = np.diff(e, axis=0) / (x[4:] - x[:-4])[:, None]
        return s, ds, e, f

    def _approxder(self, diffs):
        # Derivatives of the Newton interpolating polynomials through each node and the
        # nodes to its right, or, for the last few nodes, through the last nodes
        s, ds, e, f = diffs
        x = self._knots[:, None]
        n = self.n
        d = np.empty((n, s.shape[1]))

        if self.approx_order == 'cubic':
            at = self._offsets(0, n - 3)
            a = x[at(0)] - x[at(1)]
            b = x[at(0)] - x[at(2)]
            d[at(0)] = s[at(0)] + ds[at(0)]*a + e[at(0)]*a*b
            u = [x[n-3:] - x[n-4+i] for i in range(3)]
            d[n-3:] = s[n-4] + ds[n-4]*(u[0] + u[1]) + e[n-4]*(u[0]*u[1] + u[0]*u[2] + u[1]*u[2])
        elif self.approx_order == 'quartic':
            at = self._offsets(0, n - 4)
            a = x[at(0)] - x[at(1)]
            b = x[at(0)] - x[at(2)]
            c = x[at(0)] - x[at(3)]
            d[at(0)] = s[at(0)] + ds[at(0)]*a + e[at(0)]*a*b + f[at(0)]*a*b*c
            u = [x[n-4:] - x[n-5+i] for i in range(4)]
            d[n-4:] = s[n-5] + ds[n-5]*(u[0] + u[1]) + e[n-5]*(u[0]*u[1] + u[0]*u[2] + u[1]*u[2]) + \
                      f[n-5]*(u[0]*u[1]*u[2] + u[0]*u[1]*u[3] + u[0]*u[2]*u[3] + u[1]*u[2]*u[3])
        else:
            raise ValueError(f"Unsupported approx_order: {self.approx_order}")
        return d

    def _ttilde(self, start, stop, s, ds, emin, h):
        # Slopes at x[k] of the cubics on [x[k-1], x[k]] and [x[k], x[k+1]] (Huynh, 4.8a,b and 4.11)
        x = self._knots[:, None]
        at = self._offsets(start, stop)
        q1 = s[at(-1)] + h[at(-1)]*self._minmod(ds[at(-2)] + emin[at(-3)]*(x[at(0)] - x[at(-2)]),
                                          ds[at(-1)] + emin[at(-2)]*(x[at(0)] - x[at(1)]))
        q2 = s[at(0)]  - h[at(0)] *self._minmod(ds[at(-1)] + emin[at(-2)]*(x[at(0)] - x[at(-1)]),
                                          ds[at(0)]  + emin[at(-1)]*(x[at(0)] - x[at(2)]))
        return self._minmod(q1, q2)

    def _tmax(self, start, stop, s, dmin, smin):
        # Bound which relaxes the monotonicity constraint near extrema (Huynh, 3.13 and 3.14)
        nder = 3
        x = self._knots[:, None]
        at = self._offsets(start, stop)
        p1 = s[at(-1)] + dmin[at(-2)]*(x[at(0)] - x[at(-1)])
        p2 = s[at(0)]  + dmin[at(-1)]*(x[at(0)] - x[at(1)])
        t = self._minmod(p1, p2)
        return t, np.sign(t) * np.maximum(nder * np.abs(smin[at(-1)]), (nder / 2) * np.abs(t))

    def _perturbder(self, d, diffs):
        nder = 3
        n = self.n
        dper = d.copy()
        s, ds, e, _ = diffs
        h = np.diff(self._knots)[:, None]
        smin = self._minmod(s[:-1], s[1:])
        dmin = self._minmod(ds[:-1], ds[1:])

        if self.mono_constraint not in ('M3', 'M4'):
            raise ValueError(f"Unsupported mono_constraint: {self.mono_constraint}")

        # Basic monotonicity constraint at the endpoints, MP constraint next to them (Huynh, section 5)
        dper[0] = self._minmod(d[0], nder * s[0])
        dper[n-1] = self._minmod(d[n-1], nder * s[n-2])
        dper[1] = self._minmod(d[1], nder * self._minmod(s[0], s[1]))
        dper[n-2] = self._minmod(d[n-2], nder*self._minmod(s[n-3], s[n-2]))

        # M3 constraint at nodes 2, ..., n-3 (for M4, only at the first and last of them)
        t, tmax = self._tmax(2, n - 2, s, dmin, smin)
        if self.mono_constraint == 'M3':
            dper[2:n-2] = self._minmod(d[2:n-2], tmax)
        else:
            dper[[2, n-3]] = self._minmod(d[[2, n-3]], tmax[[0, -1]])
            if n >= 7:
                # M4 constraint at the interior nodes 3, ..., n-4
                emin = self._minmod(e[:-1], e[1:])
                tt = self._ttilde(3, n - 3, s, ds, emin, h)
                t = (nder/2)*t[1:-1]
                bound = nder*smin[2:n-4]
                tmin = np.minimum(np.minimum(0.0, bound), np.minimum(t, tt))
                tmax = np.maximum(np.maximum(0.0, bound), np.maximum(t, tt))
                dk = d[3:n-3]
                dper[3:n-3] = dk + self._minmod(tmin - dk, tmax - dk)
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
