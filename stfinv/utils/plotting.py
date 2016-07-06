#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  plotting.py
#   Purpose:   Provide routines for plotting
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

import matplotlib.pyplot as plt
import os
import numpy as np
from obspy.imaging.beachball import beach
import cartopy.crs as ccrs


def plot_waveforms(st_data, st_synth, arr_times, CC, CClim, dA, dT, stf, depth,
                   tensor, iteration=-1, misfit=0.0, outdir='./waveforms/'):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    nplots = len(st_data)
    nrows = int(np.sqrt(nplots)) + 1
    ncols = nplots / nrows + 1
    iplot = 0
    for tr in st_data:

        irow = np.mod(iplot, nrows)
        icol = np.int(iplot / nrows)

        normfac = max(np.abs(tr.data))

        yoffset = irow * 1.5
        xoffset = icol

        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        if CC[code] > CClim:
            ls = '-'
        else:
            ls = 'dotted'

        yvals = st_synth.select(station=tr.stats.station)[0].data / normfac
        xvals = np.linspace(0, 0.8, num=len(yvals))
        l_s, = ax.plot(xvals + xoffset,
                       yvals + yoffset,
                       color='r',
                       linestyle=ls,
                       linewidth=2)

        yvals = tr.data / normfac
        xvals = np.linspace(0, 0.8, num=len(yvals))
        l_d, = ax.plot(xvals + xoffset,
                       yvals + yoffset,
                       color='k',
                       linestyle=ls,
                       linewidth=1.5)
        ax.text(xoffset, yoffset + 0.2,
                '%s \nCC: %4.2f\ndA: %4.1f\ndT: %5.1f' % (tr.stats.station,
                                                          CC[code],
                                                          dA[code],
                                                          dT[code]),
                size=8.0, color='darkgreen')
        xvals = ((arr_times[code] / tr.times()[-1]) * 0.8 + xoffset) * \
            np.ones(2)
        ax.plot(xvals, (yoffset + 0.5, yoffset - 0.5), 'b')

        iplot += 1

    ax.legend((l_s, l_d), ('Synthetic', 'data'))
    ax.set_xlim(0, ncols * 1.2)

    if (iteration >= 0):
        ax.set_title('Waveform fits, depth: %d m, it: %d, misfit: %9.3e' %
                     (depth, iteration, misfit))

    # Plot STF
    left, bottom, width, height = [0.8, 0.2, 0.14, 0.18]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(stf)
    ax2.plot([0, 1000], [0, 0], '--')
    ax2.set_ylim((-0.2, 1.1))
    ax2.set_xlim((0.0, len(yvals)))
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot beach ball
    mt = [tensor.m_rr, tensor.m_tt, tensor.m_pp,
          tensor.m_rt, tensor.m_rp, tensor.m_tp]
    b = beach(mt, width=50, linewidth=1, facecolor='r',
              xy=(10, 0.5), axes=ax2)
    ax2.add_collection(b)

    # Plot Map
    plot_map(st_synth, CC, -30, 50, fig=fig,
             rect=[0.7, 0.4, 0.25, 0.30])

    outfile = os.path.join(outdir, 'waveforms_it_%d.png' % iteration)
    fig.savefig(outfile, format='png')
    plt.close(fig)


def plot_map(st_synth, values, central_longitude, central_latitude,
             colormap='viridis', fig=None, rect=[1.0, 1.0, 1.0, 1.0]):

    # Create array of latitudes, longitudes, CC
    lats = []
    lons = []
    c = []
    for tr in st_synth:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        lats.append(tr.stats.sac.stla)
        lons.append(tr.stats.sac.stlo)
        c.append(values[code])

    ax = add_ortho(lats=lats, lons=lons, color=c,
                   colormap=colormap, fig=fig, rect=rect,
                   central_longitude=central_longitude,
                   central_latitude=central_latitude)

    return ax


def add_ortho(lats, lons, color, size=1e2, marker='o',
              colormap='viridis', fig=None,
              rect=[0.0, 0.0, 1.0, 1.0],
              central_longitude=8, central_latitude=50):
    if not fig:
        fig = plt.figure()

    proj = ccrs.Orthographic(central_longitude=central_longitude,
                             central_latitude=central_latitude)
    ax = fig.add_axes(rect, projection=proj)

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()

    ax.stock_img()
    ax.coastlines()
    ax.gridlines()

    plt.scatter(lats, lons, s=size, c=color, marker=marker, cmap=colormap,
                transform=ccrs.Geodetic())

    return ax
