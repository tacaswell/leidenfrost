#Copyright 2013 Thomas A Caswell
#tcaswell@uchicago.edu
#http://jfi.uchicago.edu/~tcaswell
#
#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3 of the License, or (at
#your option) any later version.
#
#This program is distributed in the hope that it will be useful, but
#WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program; if not, see <http://www.gnu.org/licenses>.

from __future__ import division

import numpy as np
import scipy.odr as sodr


class EllipseException(Exception):
    pass


# http://mathworld.wolfram.com/Ellipse.html
def gen_to_parm(p):
    '''
    Converts the parameters for the general quadratic form to the human readable parameters.

    Parameters
    ----------
    p: sequence
       (a, b, c, d, f) = p

    Returns
    -------
    new_p: tuple
        (axis_1, axis_2, theta_0, cntr_x, cntr_y)
    '''
    a, b, c, d, f = p
    g = -1

    x0 = (c*d - b*f) / (b*b - a*c)
    y0 = (a*f - b*d) / (b*b - a*c)
    ap = np.sqrt((2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)) / ((b*b - a*c) * (np.sqrt((a-c) ** 2 + 4*b*b) - (a+c))))
    bp = np.sqrt((2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)) / ((b*b - a*c) * (-np.sqrt((a-c) ** 2 + 4*b*b) - (a+c))))

    if b == 0:
        if a < c:
            t0 = 0
        elif c < a:
            t0 = (1/2) * np.pi
    else:
        if a < c:
            t0 = (1 / 2) * np.arctan(2 * b / (a - c))

        elif a > c:
            t0 = np.pi / 2 + (1 / 2) * np.arctan(2 * b / (a - c))

    return (ap, bp, t0, x0, y0)


def gen_ellipse(a, b, t, x, y, theta):
    '''Return the (x, y) coordinates for the ellipse with
    the given parameters.

    Parameters
    ----------
    a: float
        major axis
    b: float
        minor axis
    t: float
        the angle between the x-axis and the major axis
    x: float
        x coordinate of center
    y: float
        y coordinate of center
    theta: `np.ndarray` or float
        the angles to compute the coordinates at

    Returns: `np.ndarray`
        A 2xN `np.ndarray` [x_vals, y_vals]

    '''
    if a < b:
        a, b = b, a
        t = t + np.pi/2

    assert a >= b, (a, b)

    r = 1 / np.sqrt((np.cos(theta - t) ** 2) / (a * a) + (np.sin(theta - t) ** 2) / (b * b))
    return np.vstack((r * np.cos(theta) + x, r * np.sin(theta) + y))


def _e_funx(p, r):
    '''
    private helper function for ellipse fitting
    '''
    x, y = r
    a, b, c, d, f = p
    return (a * x * x) + (2 * b * x * y) + (c * y * y) + (2 * d * x) + (2 * f * y) - 1


def gap_filler(x, y, R, center, fill_density=0.1):
    """
    Returns N points to fill in a gap

    Parameters
    ----------
    x: sequence
        x = (x_left, x_right)
    y: sequence
        y = (y_left, y_right)
    R: float
        initial guess at average radius
    center: `np.ndarray`
        initial guess at the center of the ellipse
    N: int
        The number of points to return, defaults to 15

    Returns: `np.ndarray`
        A 2xN `np.ndarray` [x_vals, y_vals]
    """
    xl, xr = x
    yl, yr = y

    R2 = R * R

    a = c = 1 / R2
    b = 0
    d = -center[0] / R2
    f = -center[1] / R2

    p0 = (a, b, c, d, f)
    data = sodr.Data((np.hstack(x), np.hstack(y)), 1)
    model = sodr.Model(_e_funx, implicit=1)
    worker = sodr.ODR(data, model, p0)
    out = worker.run()
    out = worker.restart()

    ap, bp, t0, x0, y0 = gen_to_parm(out.beta)

    if np.any(np.isnan([ap, bp, t0, x0, y0])):
        #        print out.beta
        #        print ap, bp, t0, x0, y0
        raise EllipseException('parameters contain NaN')
    theta_start = np.mod(np.arctan2(yl[-1] - y0, xl[-1] - x0), 2 * np.pi)
    theta_end = np.mod(np.arctan2(yr[0] - y0, xr[0] - x0), 2 * np.pi)
    if theta_end < theta_start:
        theta_end += 2 * np.pi

    N = int((theta_end - theta_start) * fill_density * (ap + bp) / 2)
    #    print theta_start, theta_end, (ap + bp) / 2, N
    if N < 1:
        print 'gap_filler: under fill'
        N = 1

    theta_list = np.linspace(theta_start, theta_end, N + 2)[1:-1]

    return gen_ellipse(ap, bp, t0, x0, y0, theta_list)
