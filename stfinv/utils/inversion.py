#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  inversion.py
#   Purpose:   Provide routines for inversion of STF and MT
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

import os
import numpy as np
import obspy
from scipy.optimize import lsq_linear
from scipy.linalg import toeplitz
from .depth import Depth
from .iteration import Iteration
from .stream import correct_waveforms


def inversion(db, st, origin, tensor, stf,
              depth_in_m=-1, dist_min=30.0, dist_max=100.0, CClim=0.6,
              phase_list=('P', 'Pdiff'),
              pre_offset=15,
              post_offset=36.1,
              tol=1e-3, misfit='CC',
              work_dir='testinversion'):

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # Instaseis does not like sources at 0.0 km
    if depth_in_m == 0.0:
        depth_in_m += 0.01

    print('Inverting for depth %5.2fkm' % (depth_in_m * 1e-3))

    # Calculate synthetics in GRF6 format with instaseis and cut
    # time windows around the phase arrivals (out of data and GRF6 synthetics.
    # The cuts are saved on disk for the next time.
    st_data, st_synth_grf6 = st.get_synthetics(origin,
                                               db,
                                               depth_in_m=depth_in_m,
                                               out_dir=work_dir,
                                               pre_offset=pre_offset,
                                               post_offset=post_offset,
                                               dist_min=dist_min,
                                               dist_max=dist_max,
                                               phase_list=phase_list)

    st_data.filter(type='lowpass', freq=1. / db.info.dt / 4)

    # Define butterworth filter at database corner frequency
    # b, a = signal.butter(6, Wn=0.5)

    # Start values to ensure one iteration
    it = 0
    misfit_reduction = 1e8
    misfit_new = 2
    res = Depth()

    while True:
        # Get synthetics for current source solution
        st_synth = st_synth_grf6.calc_synthetic_from_grf6(st_data,
                                                          tensor=tensor)

        offset = -10.0

        if it == 0:
            freq = 1. / 10.
        else:
            freq = None

        st_data_work, st_synth_corr, st_synth_grf6_corr, CC, dT, dA = \
            correct_waveforms(st_data,
                              st_synth,
                              st_synth_grf6,
                              allow_negative_CC=False,
                              freq=freq,
                              offset=offset)

        arr_times = st_synth_corr.pick()

        nstat_used = len(st_data_work.filter_bad_waveforms(CC, CClim))

        # Calculate misfit reduction
        misfit_old = misfit_new
        if misfit == 'CC':
            misfit_new = calc_D_misfit(CC)
        elif misfit == 'L2':
            misfit_new = calc_L2_misfit(st_data_work, st_synth_corr)

        misfit_reduction = (misfit_old - misfit_new) / misfit_old
        res_it = Iteration(tensor=tensor,
                           origin=origin,
                           stf=stf,
                           CC=CC,
                           dA=dA,
                           dT=dT,
                           arr_times=arr_times,
                           it=it,
                           CClim=CClim,
                           depth=depth_in_m,
                           misfit=misfit_new,
                           st_data=st_data_work,
                           st_synth=st_synth_corr)

        # Plot result of this iteration
        plot_dir = os.path.join(work_dir,
                                'waveforms_%06dkm' % (depth_in_m))
        res_it.plot(outdir=plot_dir)

        res.append(res_it)

        print('  it: %02d, misfit: %5.3f (%8.1f pct red. %d stations)' %
              (it, misfit_new, misfit_reduction * 1e2, nstat_used))

        # Stop the inversion, if no station can be used
        if (nstat_used == 0):
            print('All stations have CC below limit (%4.2f), stopping' % CClim)
            break
        elif misfit_reduction <= tol and it > 1:
            print('Inversion seems to have converged')
            break

        # Omit the STF inversion in the first round.
        if it > 0:
            st_data_filtCC = st_data_work.filter_bad_waveforms(CC, CClim)

            st_synth_filtCC = st_synth_corr.filter_bad_waveforms(CC, CClim)
            stf = invert_STF(st_data_filtCC, st_synth_filtCC,
                             method='dampened', eps=1e-1,
                             len_stf=30.0)

        tensor = invert_MT(st_data_work.filter_bad_waveforms(CC, CClim),
                           st_synth_grf6_corr.filter_bad_waveforms(CC, CClim),
                           stf=stf,
                           outdir=os.path.join(work_dir, 'focmec'))
        it += 1

    save_fnam = os.path.join(work_dir,
                             'waveforms_%06dkm' % depth_in_m,
                             'result.npz')
    res.save(fnam=save_fnam)

    return res


def _create_Toeplitz(data, npts_stf):
    # Desired dimensions: len(data) + npts_stf - 1 x npts_stf
    nrows = npts_stf + len(data) - 1
    data_col = data
    first_col = np.r_[data_col,
                      np.zeros(nrows - len(data_col))]

    ncols = npts_stf
    first_row = np.r_[data[0], np.zeros(ncols - 1)]
    return toeplitz(first_col, first_row)


def _create_Toeplitz_mult(stream, npts_stf, fix_npts=True):
    nstat = len(stream)

    if fix_npts:
        npts = stream[0].stats.npts
    else:
        npts = stream[0].stats.npts + npts_stf

    G = np.zeros((nstat * npts, npts_stf))
    for istat in range(0, nstat):
        G[istat * npts:(istat + 1) * npts][:] = \
            _create_Toeplitz(stream[istat].data,
                             npts_stf)[0:npts, :]
    return G


def _create_matrix_STF_inversion(st_data, st_synth, npts_stf):
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
    G = _create_Toeplitz_mult(st_synth, npts_stf, fix_npts=True)
    # GTG = np.matmul(G.transpose, G)
    # GTGmean = np.diag(GTG).mean()
    # J = np.diag(np.linspace(0, GTGmean, GTG.shape[0]))
    # A = np.matmul(np.inv(np.matmul(GTG, J))

    return d, G


def invert_STF(st_data, st_synth, method='bound_lsq', len_stf=None, eps=1e-3):
    # print('Using %d stations for STF inversion' % len(st_data))
    # for tr in st_data:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.plot(tr.data, label='data')
    #     ax.plot(st_synth.select(station=tr.stats.station,
    #                             network=tr.stats.network,
    #                             location=tr.stats.location)[0].data,
    #             label='synth')
    #     ax.legend()
    #     fig.savefig('%s.png' % tr.stats.station)
    #     plt.close(fig)

    # Calculate number of samples for STF:
    if len_stf:
        npts_stf = int(len_stf * st_data[0].stats.delta)
    else:
        npts_stf = st_data[0].stats.npts

    d, G = _create_matrix_STF_inversion(st_data, st_synth, npts_stf)

    if method == 'bound_lsq':
        m = lsq_linear(G, d, (-0.1, 1.1))
        stf = np.r_[m.x[(len(m.x) - 1) / 2:], m.x[0:(len(m.x) - 1) / 2]]

    elif method == 'lsq':
        stf, residual, rank, s = np.linalg.lstsq(G, d)

    elif method == 'dampened':
        # J from eq.3 in Sigloch (2006) to dampen later part of STFs
        GTG = np.matmul(G.T, G)
        diagGmean = np.mean(np.diag(GTG))
        J = np.diag(np.linspace(0, diagGmean, G.shape[1]))
        Ginv = np.matmul(np.linalg.inv(GTG + eps * J), G.T)
        stf = np.matmul(Ginv, d)

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
        tr.data = np.convolve(tr.data, stf)[0:tr.stats.npts]

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


def calc_D_misfit(CCs):
    CC = []
    for key, value in CCs.items():
        CC.append(1. - value)

    return np.mean(CC)


def calc_L2_misfit(st_a, st_b):
    L2 = 0.0
    for tr_a in st_a:
        tr_b = st_b.select(station=tr_a.stats.station,
                           network=tr_a.stats.network,
                           location=tr_a.stats.location)[0]
        L2 += np.sum(tr_a.data - tr_b.data)**2
        # L2 /= np.sum(tr_b.data)**2
    return np.sqrt(L2)
