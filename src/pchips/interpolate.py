import numpy as np

class PchipInterpolator:
    """
    PchipInterpolator: Piecewise Cubic Hermite Interpolating Polynomial.

    This class mimics the behavior of `scipy.interpolate.PchipInterpolator`
    but uses the specific algorithms from the MATLAB repository.
    """

    def __init__(self, x, y, approx_order='cubic', mono_constraint='M3'):
        """
        Initializes the interpolator.

        Args:
            x (np.ndarray): The knot vector.
            y (np.ndarray): The data vector.
            approx_order (str): The order of the derivative approximation formula.
                                Supported: 'cubic' (default), 'quartic'.
            mono_constraint (str): The monotonicity constraint type.
                                   Supported: 'M3' (default), 'M4'.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y must be numpy arrays.")
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-dimensional.")
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")
        if x.shape[0] < 5:
            raise ValueError("There should be at least five data points.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("x and y must contain finite values.")

        self.approx_order = approx_order
        self.mono_constraint = mono_constraint

        # Sort x and y
        idx = np.argsort(x)
        self.x = x[idx]
        self.y = y[idx]
        
        h = np.diff(self.x)
        if np.any(h <= 0):
            raise ValueError("The data abscissae should be distinct and increasing.")

        self.n = len(self.x)
        
        d = self._approxder()
        d = self._perturbder(d)
        self._compute_coeffs(d)

    def _vecdiffs(self):
        y_temp = self.y.copy()
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

    def _approxder(self):
        d = np.zeros(self.n)
        s, ds, e, f = self._vecdiffs()

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

    def _perturbder(self, d):
        nder = 3
        dper = d.copy()
        s, ds, e, f = self._vecdiffs()
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
            dper[1] = np.sign(d[1]) * np.minimum(np.abs(d[1]), nder * np.abs(s[0]))
            dper[1] = np.sign(dper[1]) * np.minimum(np.abs(dper[1]), nder * np.abs(s[1]))
            dper[self.n-2] = np.sign(d[self.n-2]) * np.minimum(np.abs(d[self.n-2]), nder*np.abs(s[self.n-3]))
            dper[self.n-2] = np.sign(dper[self.n-2]) * np.minimum(np.abs(dper[self.n-2]), nder*np.abs(s[self.n-2]))
            t = self._minmod(s[1] + dmin[0]*(self.x[2] - self.x[1]), s[2] + dmin[1]*(self.x[2] - self.x[3]))
            tmax = np.sign(t)*max(nder*abs(smin[1]), (nder/2)*abs(t))
            dper[2] = self._minmod(d[2], tmax)
            t = self._minmod(s[self.n-4] + dmin[self.n-5]*(self.x[self.n-3] - self.x[self.n-4]), s[self.n-3] + dmin[self.n-4]*(self.x[self.n-3] - self.x[self.n-2]))
            tmax = np.sign(t)*max(nder*abs(smin[self.n-4]), (nder/2)*abs(t))
            dper[self.n-3] = self._minmod(d[self.n-3], tmax)
            for k in range(3, self.n - 3):
                p1 = s[k-1] + dmin[k-2]*(self.x[k] - self.x[k-1])
                p2 = s[k]   + dmin[k-1]*(self.x[k] - self.x[k+1])
                q1 = s[k-1] - h[k-1]*self._minmod(ds[k-2] + emin[k-3]*(self.x[k-1] - self.x[k-2]),
                                            ds[k-1] + emin[k-2]*(self.x[k-1] - self.x[k+1]))
                q2 = s[k]   - h[k]  *self._minmod(ds[k-1] + emin[k-2]*(self.x[k]   - self.x[k-1]),
                                            ds[k]   + emin[k-1]*(self.x[k]   - self.x[k+2]))
                t  = self._minmod(p1,p2)
                tt = self._minmod(q1,q2)
                vec = np.array([0, nder*smin[k-1], (nder/2)*t, tt])
                min_vec = np.min(vec)
                max_vec = np.max(vec)
                dper[k] = d[k] + self._minmod(min_vec - d[k], max_vec - d[k])
        else:
            raise ValueError(f"Unsupported mono_constraint: {self.mono_constraint}")
        return dper

    def _compute_coeffs(self, d):
        h = np.diff(self.x)
        delta = np.diff(self.y) / h
        self.coeffs = np.zeros((self.n - 1, 4))
        c3 = (d[:-1] - 2*delta + d[1:]) / h**2
        c2 = (3*delta - 2*d[:-1] - d[1:]) / h
        self.coeffs[:, 0] = c3
        self.coeffs[:, 1] = c2
        self.coeffs[:, 2] = d[:-1]
        self.coeffs[:, 3] = self.y[:-1]

    def __call__(self, u):
        is_scalar = not isinstance(u, np.ndarray)
        if is_scalar:
            u = np.array([u])
        indices = np.searchsorted(self.x, u, side='right') - 1
        indices = np.clip(indices, 0, self.n - 2)
        s = u - self.x[indices]
        c3 = self.coeffs[indices, 0]
        c2 = self.coeffs[indices, 1]
        c1 = self.coeffs[indices, 2]
        c0 = self.coeffs[indices, 3]
        v = c0 + s * (c1 + s * (c2 + s * c3))
        return v[0] if is_scalar else v
