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
import scipy.signal
import numpy as np

"""
module to hold a collection of figure panel making code
"""


def plot_average_height(RM, h5_backend, ax,
                        window=13, order=3, h_scale=.6328,
                        label='height', c='k', **kwargs):
    """
    Plots the average height of the rim vs time smoothed with
    savgol filter (13, 3)

    Parameters
    ----------
    RM : region_map
       The source of height data

    h5_backend : hdfbackend
       The source of the rest of the data


    Returns
    -------
    list : list of artists added
    """

    rs_image = RM.resampled_height
    ax.set_ylabel('$\\overline{h}$ [\\textmu m]')
    ax.set_xlabel('$\\tau$ [s]')
    ln = ax.plot(np.arange(rs_image.shape[1])/h5_backend.frame_rate,
            h_scale * np.mean(scipy.signal.savgol_filter(
                rs_image, window, order, deriv=0, axis=1), axis=0),
            c=c, label=label, **kwargs)

    return ln


def circumference_vs_time_ax(self, ax, t_scale=1, c_scale=1,
                             t_offset=0, f_slice=None, **kwargs):
    '''
    Plots the circumference vs time onto the given ax.

    Extra `**kwargs` are passed to `plot`

    Parameters
    ----------
    ax : `matplotlib.Axes`
        The axes to plot the data to
    t_scale : float
        Scale factor to apply the frame number for plotting.

        The displayed time will be
        :math:`(frame_number - t_offset) * t_scale`
    t_offset : int
        off set to apply to the frame number before scaling.

        The displayed time will be
        :math:`(frame_number - t_offset) * t_scale`

    f_slice : `slice` or None
        which frames to plot

    '''
    old_params = self.prams
    self.prams = (False, False)

    if f_slice is None:
        f_slice = slice(None)

    t = ((np.arange(*f_slice.indices(len(self))) - t_offset) *
          t_scale / self.frame_rate)
    c = (np.array([mbe.curve.circ for mbe in self[f_slice]]) *
          c_scale * self.calibration_value * 1e-3)

    self.prams = old_params

    return ax.plot(t, c, **kwargs)
