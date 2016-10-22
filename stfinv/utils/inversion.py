#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  inversion.py
#   Purpose:   Provide routines for inversion of STF and MT
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

import numpy as np
import obspy
from scipy.optimize import lsq_linear
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt


def _create_Toeplitz(data):
    npts = len(data)
    padding = np.zeros((npts) / 2 + 1)

    if np.mod(npts, 2) == 0:
        # even number of elements
        start = (npts) / 2 - 1
        first_col = np.r_[data[start:-1], padding[0:-1]]
    else:
        # odd number of elements
        start = (npts) / 2
        first_col = np.r_[data[start:-1], padding]

    first_row = np.r_[data[start:0:-1], padding]
    return toeplitz(first_col, first_row)


def _create_Toeplitz_mult(stream):
    nstat = len(stream)
    npts = stream[0].stats.npts
    G = np.zeros((nstat * npts, npts))
    # print('G:')
    for istat in range(0, nstat):
        # print(istat, stream[istat].stats.station)
        G[istat * npts:(istat + 1) * npts][:] = \
            _create_Toeplitz(stream[istat].data)
    return G


def _create_matrix_STF_inversion(st_data, st_synth):
    # Create matrix for STF inversion:
    npts = st_synth[0].stats.npts
    nstat = len(st_data)

    # Check number of traces in input streams
    if (nstat != len(st_synth)):
        raise IndexError('len(st_synth) has to be len(st_data)')

    # Create data vector (all waveforms concatenated)
    d = np.zeros(npts * nstat)
    # print('d:')
    for istat in range(0, nstat):
        # print(istat, st_data[istat].stats.station)
        d[istat * npts:(istat + 1) * npts] = st_data[istat].data[0:npts]

    # Create G-matrix
    G = _create_Toeplitz_mult(st_synth)
    # GTG = np.matmul(G.transpose, G)
    # GTGmean = np.diag(GTG).mean()
    # J = np.diag(np.linspace(0, GTGmean, GTG.shape[0]))
    # A = np.matmul(np.inv(np.matmul(GTG, J))

    return d, G


def invert_STF(st_data, st_synth, method='bound_lsq'):
    print('Using %d stations for STF inversion' % len(st_data))
    for tr in st_data:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(tr.data, label='data')
        ax.plot(st_synth.select(station=tr.stats.station,
                                network=tr.stats.network,
                                location=tr.stats.location)[0].data,
                label='synth')
        ax.legend()
        fig.savefig('%s.png' % tr.stats.station)
        plt.close(fig)

    # st_data.write('data.mseed', format='mseed')
    # st_synth.write('synth.mseed', format='mseed')

    d, G = _create_matrix_STF_inversion(st_data, st_synth)

    if method == 'bound_lsq':
        m = lsq_linear(G, d, (-0.1, 1.1))
        stf = np.r_[m.x[(len(m.x) - 1) / 2:], m.x[0:(len(m.x) - 1) / 2]]
    elif method == 'lsq':
        stf, residual, rank, s = np.linalg.lstsq(G, d)
    else:
        raise ValueError('method %s unknown' % method)

    return stf


def _create_matrix_MT_inversion(st_data, st_grf6):
    """
    d, G = create_matrix_MT_inversion(st_data, st_grf6):

    Create data vector d and sensitivity matrix G for the MT inversion.

    Parameters
    ----------
    st_data : obspy.Stream
        Stream with N (measured) waveforms

    st_grf6 : obspy.Stream
        Stream with 6xN synthetic Green's functions, which should be corrected
        for time shift and amplitude errors in the data.

    Returns
    -------
    d : np.array
        Data vector with shape (N * npts), where N is the number of common
        stations in st_data and st_grf6.

    G : np.array
        Data vector with shape (6 x N * npts), where N is the number of common
        stations in st_data and st_grf6.

    """

    if not st_grf6[0].stats.npts == st_data[0].stats.npts:
        raise IndexError('Len of data (%d) and synthetics (%d) is not equal)' %
                         (st_grf6[0].stats.npts, st_data[0].stats.npts))

    # Create matrix for MT inversion:
    npts = st_grf6[0].stats.npts

    nstat = len(st_data)

    # Check number of traces in input streams
    if (nstat * 6 != len(st_grf6)):
        raise IndexError('len(st_grf6) has to be 6*len(st_data)')

    # Create data vector (all waveforms concatenated)
    d = np.zeros((npts) * nstat)
    for istat in range(0, nstat):
        d[istat * npts:(istat + 1) * npts] = st_data[istat].data[0:npts]

    # Create G-matrix
    G = np.zeros((6, npts * nstat))

    channels = ['MTT', 'MPP', 'MRR', 'MTP', 'MRT', 'MRP']
    for icomp in range(0, 6):
        for istat in range(0, nstat):
            G[icomp][istat * npts:(istat + 1) * npts] = \
                st_grf6.select(channel=channels[icomp])[istat].data[0:npts]
    return d, G


def invert_MT(st_data, st_grf6, stf=[1], outdir='focmec'):
    """
    tens_new = invert_MT(st_data, st_grf6, dA, tens_orig):

    Invert for a new moment tensor using the data stream st_data and the
    grf6 stream st_grf6.

    Parameters
    ----------
    st_data : obspy.Stream
        Stream with N (measured) waveforms

    st_grf6 : obspy.Stream
        Stream with 6xN synthetic Green's functions, which should be corrected
        for time shift and amplitude errors in the data.

    stf : np.array
        Normalized source time function (slip rate function) with the
        same sampling rate as the streams.

    outdir : String
        Path in which to write Beachballs for each iteration


    Returns
    -------
    tens_new : obspy.core.event.Tensor
        Updated moment tensor

    """

    # Create working copy
    st_grf6_work = st_grf6.copy()

    # Convolve with STF
    for tr in st_grf6_work:
        tr.data = np.convolve(tr.data, stf, mode='same')[0:tr.stats.npts]

    d, G = _create_matrix_MT_inversion(st_data, st_grf6_work)

    m, residual, rank, s = np.linalg.lstsq(G.T, d)

    # Order in m:
    # ['MXX', 'MYY', 'MZZ', 'MXY', 'MXZ', 'MYZ']
    #   mtt,   mpp,   mrr,   mtp    mrt    mrp
    tens_new = obspy.core.event.Tensor(m_tt=m[0],
                                       m_pp=m[1],
                                       m_rr=m[2],
                                       m_tp=m[3],
                                       m_rt=m[4],
                                       m_rp=m[5])
    return tens_new
