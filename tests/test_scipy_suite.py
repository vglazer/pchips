# SciPy's own PchipInterpolator tests, run against pchips.
#
# Adapted from scipy/interpolate/tests/test_polyint.py (SciPy 1.16.1): only the
# imports and assertion helpers are changed, and the tests that check properties
# or values of Fritsch and Butland's method which Huynh's method does not share are
# marked as expected failures.
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from pchips import PchipInterpolator, pchip, pchip_interpolate


def xp_assert_close(actual, desired, **kwargs):
    assert_allclose(actual, desired, **kwargs)


def check_shape(interpolator_cls, x_shape, y_shape, deriv_shape=None, axis=0,
                extra_args=None):
    if extra_args is None:
        extra_args = {}
    rng = np.random.RandomState(1234)

    x = [-1, 0, 1, 2, 3, 4]
    s = list(range(1, len(y_shape)+1))
    s.insert(axis % (len(y_shape)+1), 0)
    y = rng.rand(*((6,) + y_shape)).transpose(s)

    xi = np.zeros(x_shape)
    yi = interpolator_cls(x, y, axis=axis, **extra_args)(xi)

    target_shape = ((deriv_shape or ()) + y.shape[:axis]
                    + x_shape + y.shape[axis:][1:])
    assert yi.shape == target_shape

    # check it works also with lists
    if x_shape and y.size > 0:
        interpolator_cls(list(x), list(y), axis=axis,
                         **extra_args)(list(xi))

    # check also values
    if xi.size > 0 and deriv_shape is None:
        bs_shape = y.shape[:axis] + (1,)*len(x_shape) + y.shape[axis:][1:]
        yv = y[((slice(None,),)*(axis % y.ndim)) + (1,)]
        yv = yv.reshape(bs_shape)

        yi, y = np.broadcast_arrays(yi, yv)
        xp_assert_close(yi, y)


SHAPES = [(), (0,), (1,), (6, 2, 5)]


def test_shapes():
    for s1 in SHAPES:
        for s2 in SHAPES:
            for axis in range(-len(s2), len(s2)):
                check_shape(pchip, s1, s2, None, axis)


def test_deriv_shapes():
    def pchip_deriv(x, y, axis=0):
        return pchip(x, y, axis).derivative()

    def pchip_deriv2(x, y, axis=0):
        return pchip(x, y, axis).derivative(2)

    def pchip_antideriv(x, y, axis=0):
        return pchip(x, y, axis).antiderivative()

    def pchip_antideriv2(x, y, axis=0):
        return pchip(x, y, axis).antiderivative(2)

    def pchip_deriv_inplace(x, y, axis=0):
        class P(PchipInterpolator):
            def __call__(self, x):
                return PchipInterpolator.__call__(self, x, 1)
        return P(x, y, axis)

    for ip in [pchip_deriv, pchip_deriv2, pchip_deriv_inplace,
               pchip_antideriv, pchip_antideriv2]:
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    check_shape(ip, s1, s2, (), axis)


class TestPCHIP:
    def _make_random(self, npts=20):
        rng = np.random.RandomState(1234)
        xi = np.sort(rng.random(npts))
        yi = rng.random(npts)
        return pchip(xi, yi), xi, yi

    @pytest.mark.xfail(strict=True, reason="Huynh's method may overshoot near strict local extrema")
    def test_overshoot(self):
        # PCHIP should not overshoot
        p, xi, yi = self._make_random()
        for i in range(len(xi)-1):
            x1, x2 = xi[i], xi[i+1]
            y1, y2 = yi[i], yi[i+1]
            if y1 > y2:
                y1, y2 = y2, y1
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            assert ((y1 <= yp + 1e-15) & (yp <= y2 + 1e-15)).all()

    def test_monotone(self):
        # PCHIP should preserve monotonicty
        p, xi, yi = self._make_random()
        for i in range(len(xi)-1):
            x1, x2 = xi[i], xi[i+1]
            y1, y2 = yi[i], yi[i+1]
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            assert ((y2-y1) * (yp[1:] - yp[:1]) > 0).all()

    def test_cast(self):
        # regression test for integer input data, see gh-3453
        data = np.array([[0, 4, 12, 27, 47, 60, 79, 87, 99, 100],
                         [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55]])
        xx = np.arange(100)
        curve = pchip(data[0], data[1])(xx)

        data1 = data * 1.0
        curve1 = pchip(data1[0], data1[1])(xx)

        xp_assert_close(curve, curve1, atol=1e-14, rtol=1e-14)

    @pytest.mark.xfail(strict=True, reason="Checks Fritsch and Butland's values from NAG, which Huynh's method does not reproduce")
    def test_nag(self):
        # Example from NAG C implementation,
        # http://nag.com/numeric/cl/nagdoc_cl25/html/e01/e01bec.html
        # suggested in gh-5326 as a smoke test for the way the derivatives
        # are computed (see also gh-3453)
        dataStr = '''
          7.99   0.00000E+0
          8.09   0.27643E-4
          8.19   0.43750E-1
          8.70   0.16918E+0
          9.20   0.46943E+0
         10.00   0.94374E+0
         12.00   0.99864E+0
         15.00   0.99992E+0
         20.00   0.99999E+0
        '''
        data = np.loadtxt(io.StringIO(dataStr))
        pch = pchip(data[:,0], data[:,1])

        resultStr = '''
           7.9900       0.0000
           9.1910       0.4640
          10.3920       0.9645
          11.5930       0.9965
          12.7940       0.9992
          13.9950       0.9998
          15.1960       0.9999
          16.3970       1.0000
          17.5980       1.0000
          18.7990       1.0000
          20.0000       1.0000
        '''
        result = np.loadtxt(io.StringIO(resultStr))
        xp_assert_close(result[:,1], pch(result[:,0]), rtol=0., atol=5e-5)

    def test_endslopes(self):
        # this is a smoke test for gh-3453: PCHIP interpolator should not
        # set edge slopes to zero if the data do not suggest zero edge derivatives
        x = np.array([0.0, 0.1, 0.25, 0.35])
        y1 = np.array([279.35, 0.5e3, 1.0e3, 2.5e3])
        y2 = np.array([279.35, 2.5e3, 1.50e3, 1.0e3])
        for pp in (pchip(x, y1), pchip(x, y2)):
            for t in (x[0], x[-1]):
                assert pp(t, 1) != 0

    def test_all_zeros(self):
        x = np.arange(10)
        y = np.zeros_like(x)

        # this should work and not generate any warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            pch = pchip(x, y)

        xx = np.linspace(0, 9, 101)
        assert all(pch(xx) == 0.)

    def test_two_points(self):
        # regression test for gh-6222: pchip([0, 1], [0, 1]) fails because
        # it tries to use a three-point scheme to estimate edge derivatives,
        # while there are only two points available.
        # Instead, it should construct a linear interpolator.
        x = np.linspace(0, 1, 11)
        p = pchip([0, 1], [0, 2])
        xp_assert_close(p(x), 2*x, atol=1e-15)

    def test_pchip_interpolate(self):
        assert_array_almost_equal(
            pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=1),
            np.asarray([1.]))

        assert_array_almost_equal(
            pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=0),
            np.asarray([3.5]))

        assert_array_almost_equal(
            np.asarray(pchip_interpolate([1, 2, 3], [4, 5, 6], [0.5], der=[0, 1])),
            np.asarray([[3.5], [1]]))

    def test_roots(self):
        # regression test for gh-6357: .roots method should work
        p = pchip([0, 1], [-1, 1])
        r = p.roots()
        xp_assert_close(r, np.asarray([0.5]))


class TestZeroSizeArrays:
    # regression tests for gh-17241 : CubicSpline et al must not segfault
    # when y.size == 0

    @pytest.mark.parametrize('y', [np.zeros((10, 0, 5)),
                                   np.zeros((10, 5, 0))])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('cls', [PchipInterpolator])
    def test_zero_size_2(self, cls, y, axis):
        x = np.arange(10)
        xval = np.arange(3)

        obj = cls(x, y)
        assert obj(xval).size == 0
        assert obj(xval).shape == xval.shape + y.shape[1:]

        # Also check with an explicit non-default axis
        yt = np.moveaxis(y, 0, axis)  # (10, 0, 5) --> (0, 10, 5) if axis=1 etc

        obj = cls(x, yt, axis=axis)
        sh = yt.shape[:axis] + (xval.size, ) + yt.shape[axis+1:]
        assert obj(xval).size == 0
        assert obj(xval).shape == sh
