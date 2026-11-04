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
            # Compute the derivatives one column at a time
            columns = y.reshape(self.n, int(np.prod(y.shape[1:])))
            dydx = np.empty_like(columns)
            for j in range(columns.shape[1]):
                d = self._approxder(columns[:, j])
                dydx[:, j] = self._perturbder(d, columns[:, j])
            dydx = dydx.reshape(y.shape)

        super().__init__(x, y, dydx, axis=0, extrapolate=extrapolate)
        self.axis = axis

    def _vecdiffs(self, y):
        y_temp = y.copy()
        diffs = np.zeros((4, self.n - 1))
        for i in range(4):
            if i > 0:
                y_temp = diffs[i-1, :self.n-i]
            h = self.x[i+1:] - self.x[:-(i+1)]
            diffs[i, :self.n-1-i] = (y_temp[1:self.n-i] - y_temp[:self.n-1-i]) / h

        s = diffs[0, :]
        ds = diffs[1, :self.n-2]
        e = diffs[2, :self.n-3]
        f = diffs[3, :self.n-4]
        return s, ds, e, f

    def _approxder(self, y):
        d = np.zeros(self.n)
        s, ds, e, f = self._vecdiffs(y)

        if self.approx_order == 'cubic':
            for k in range(self.n - 3):
                d[k] = s[k] + ds[k]*(self.x[k] - self.x[k+1]) + \
                       e[k]*(self.x[k]**2 + self.x[k+1]*self.x[k+2] - self.x[k]*self.x[k+1] - self.x[k]*self.x[k+2])
            w1 = 3*e[self.n-4]
            w2 = 2*(ds[self.n-4] - e[self.n-4]*(self.x[self.n-4] + self.x[self.n-3] + self.x[self.n-2]))
            rest = s[self.n-4] - ds[self.n-4]*(self.x[self.n-4] + self.x[self.n-3]) + \
                   e[self.n-4]*(self.x[self.n-4]*self.x[self.n-3] + self.x[self.n-4]*self.x[self.n-2] + self.x[self.n-3]*self.x[self.n-2])
            for k in range(self.n - 3, self.n):
                 d[k] = w1*self.x[k]**2 + w2*self.x[k] + rest
        elif self.approx_order == 'quartic':
            for k in range(self.n - 4):
                d[k] = s[k] + ds[k]*(self.x[k] - self.x[k+1]) + \
                       e[k]*(self.x[k]*(self.x[k] - self.x[k+1] - self.x[k+2]) + self.x[k+1]*self.x[k+2]) + \
                       f[k]*(self.x[k]**2*(self.x[k] - self.x[k+1] - self.x[k+2] - self.x[k+3]) + \
                             self.x[k]*(self.x[k+1]*self.x[k+2] + self.x[k+1]*self.x[k+3] + self.x[k+2]*self.x[k+3]) - \
                             self.x[k+1]*self.x[k+2]*self.x[k+3])
            w1 = 4*f[self.n-5]
            w2 = 3*(e[self.n-5] - f[self.n-5]*(self.x[self.n-5] + self.x[self.n-4] + self.x[self.n-3] + self.x[self.n-2]))
            w3 = 2*(ds[self.n-5] - e[self.n-5]*(self.x[self.n-5] + self.x[self.n-4] + self.x[self.n-3]) + \
                    f[self.n-5]*(self.x[self.n-5]*(self.x[self.n-4] + self.x[self.n-3] + self.x[self.n-2]) + \
                    self.x[self.n-4]*self.x[self.n-3] + self.x[self.n-4]*self.x[self.n-2] + self.x[self.n-3]*self.x[self.n-2]))
            rest = s[self.n-5] - ds[self.n-5]*(self.x[self.n-5] + self.x[self.n-4]) + \
                   e[self.n-5]*(self.x[self.n-5]*self.x[self.n-4] + self.x[self.n-5]*self.x[self.n-3] + self.x[self.n-4]*self.x[self.n-3]) - \
                   f[self.n-5]*(self.x[self.n-4]*self.x[self.n-3]*self.x[self.n-2] + \
                           self.x[self.n-5]*(self.x[self.n-4]*self.x[self.n-3] + self.x[self.n-4]*self.x[self.n-2] + self.x[self.n-3]*self.x[self.n-2]))
            for k in range(self.n - 4, self.n):
                d[k] = w1*self.x[k]**3 + w2*self.x[k]**2 + w3*self.x[k] + rest
        else:
            raise ValueError(f"Unsupported approx_order: {self.approx_order}")
        return d

    def _minmod(self, a, b):
        return (np.sign(a) + np.sign(b)) / 2 * np.minimum(np.abs(a), np.abs(b))

    def _ttilde(self, k, s, ds, emin, h):
        # Slopes at x[k] of the cubics on [x[k-1], x[k]] and [x[k], x[k+1]] (Huynh, 4.8a,b and 4.11)
        q1 = s[k-1] + h[k-1]*self._minmod(ds[k-2] + emin[k-3]*(self.x[k]   - self.x[k-2]),
                                    ds[k-1] + emin[k-2]*(self.x[k]   - self.x[k+1]))
        q2 = s[k]   - h[k]  *self._minmod(ds[k-1] + emin[k-2]*(self.x[k]   - self.x[k-1]),
                                    ds[k]   + emin[k-1]*(self.x[k]   - self.x[k+2]))
        return self._minmod(q1, q2)

    def _perturbder(self, d, y):
        nder = 3
        dper = d.copy()
        s, ds, e, f = self._vecdiffs(y)
        h = np.diff(self.x)
        smin = np.array([self._minmod(s[k], s[k+1]) for k in range(self.n - 2)])
        dmin = np.array([self._minmod(ds[k], ds[k+1]) for k in range(self.n - 3)])

        if self.mono_constraint == 'M3':
            dper[0] = self._minmod(d[0], nder * s[0])
            dper[self.n-1] = self._minmod(d[self.n-1], nder * s[self.n-2])
            dper[1] = self._minmod(d[1], nder * self._minmod(s[0], s[1]))
            dper[self.n-2] = self._minmod(d[self.n-2], nder*self._minmod(s[self.n-3], s[self.n-2]))
            for k in range(2, self.n - 2):
                p1 = s[k-1] + dmin[k-2]*(self.x[k] - self.x[k-1])
                p2 = s[k]   + dmin[k-1]*(self.x[k] - self.x[k+1])
                t = self._minmod(p1, p2)
                tmax = np.sign(t) * np.maximum(nder * np.abs(smin[k-1]), (nder / 2) * np.abs(t))
                dper[k] = self._minmod(d[k], tmax)
        elif self.mono_constraint == 'M4':
            emin = np.array([self._minmod(e[k], e[k+1]) for k in range(self.n - 4)])
            dper[0] = self._minmod(d[0], nder * s[0])
            dper[self.n-1] = self._minmod(d[self.n-1], nder * s[self.n-2])
            # MP constraint next to the endpoints (Huynh, section 5)
            dper[1] = self._minmod(d[1], nder * self._minmod(s[0], s[1]))
            dper[self.n-2] = self._minmod(d[self.n-2], nder*self._minmod(s[self.n-3], s[self.n-2]))
            t = self._minmod(s[1] + dmin[0]*(self.x[2] - self.x[1]), s[2] + dmin[1]*(self.x[2] - self.x[3]))
            tmax = np.sign(t)*max(nder*abs(smin[1]), (nder/2)*abs(t))
            dper[2] = self._minmod(d[2], tmax)
            t = self._minmod(s[self.n-4] + dmin[self.n-5]*(self.x[self.n-3] - self.x[self.n-4]), s[self.n-3] + dmin[self.n-4]*(self.x[self.n-3] - self.x[self.n-2]))
            tmax = np.sign(t)*max(nder*abs(smin[self.n-4]), (nder/2)*abs(t))
            dper[self.n-3] = self._minmod(d[self.n-3], tmax)
            for k in range(3, self.n - 3):
                p1 = s[k-1] + dmin[k-2]*(self.x[k] - self.x[k-1])
                p2 = s[k]   + dmin[k-1]*(self.x[k] - self.x[k+1])
                t  = self._minmod(p1,p2)
                tt = self._ttilde(k, s, ds, emin, h)
                vec = np.array([0, nder*smin[k-1], (nder/2)*t, tt])
                min_vec = np.min(vec)
                max_vec = np.max(vec)
                dper[k] = d[k] + self._minmod(min_vec - d[k], max_vec - d[k])
        else:
            raise ValueError(f"Unsupported mono_constraint: {self.mono_constraint}")
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
