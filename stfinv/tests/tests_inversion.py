#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  tests_inversion.py
#   Purpose:   Test routines for the inversion routines
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

from stfinv.utils.inversion import invert_MT
import obspy
import numpy as np
import numpy.testing as npt


def test_invert_MT():

    # db = instaseis.open_db('syngine://prem_a_20s')

    # evlat = 50.0
    # evlon = 8.0

    # reclat = 10.0
    # reclon = 10.0

    # evdepth = 10.0

    # # gps2dist_azimuth is not optimal here, since we need the geographical
    # # coordinates, and not WGS84. Try to  around it by switching ellipticity
    # # off.
    # distance, azi, bazi = gps2dist_azimuth(evlat, evlon,
    #                                        reclat, reclon,
    #                                        f=0.0)

    # # The following functions expect distance in degrees
    # km2deg = 360.0 / (2 * np.pi * 6378137.0)
    # distance *= km2deg

    # # Define Moment tensor
    # tensor = obspy.core.event.Tensor(m_rr=1e20, m_pp=-2e20, m_tt=0.5e20,
    #                                  m_rt=5e19, m_rp=-7e19, m_tp=-1e20)

    # # Get a reference trace with normal instaseis
    # rec = instaseis.Receiver(latitude=reclat, longitude=reclon)

    # src = instaseis.Source(latitude=evlat, longitude=evlon,
    #                        m_rr=tensor.m_rr,
    #                        m_tt=tensor.m_tt,
    #                        m_pp=tensor.m_pp,
    #                        m_tp=tensor.m_tp,
    #                        m_rt=tensor.m_rt,
    #                        m_rp=tensor.m_rp,
    #                        depth_in_m=evdepth)

    # st_ref = db.get_seismograms(src, rec, dt=0.1, components='Z')

    # get Greens functions
    # gf_synth = db.get_greens_function(epicentral_distance_in_degree=distance,
    #                                   source_depth_in_m=evdepth,
    #                                   dt=0.1)

    # Convert to GRF6 format
    # st_grf6 = stfinv.seiscomp_to_moment_tensor(gf_synth,
    #                                            azimuth=azi,
    #                                            scalmom=1,
    #                                            stats=st_ref[0].stats)

    st = obspy.Stream()

    tr = obspy.Trace(data=np.array([1, 0, 0, 0, 0, 0]))
    tr.stats['channel'] = 'MRR'
    st.append(tr)

    tr = obspy.Trace(data=np.array([0, 1, 0, 0, 0, 0]))
    tr.stats['channel'] = 'MTT'
    st.append(tr)

    tr = obspy.Trace(data=np.array([0, 0, 1, 0, 0, 0]))
    tr.stats['channel'] = 'MPP'
    st.append(tr)

    tr = obspy.Trace(data=np.array([0, 0, 0, 1, 0, 0]))
    tr.stats['channel'] = 'MRT'
    st.append(tr)

    tr = obspy.Trace(data=np.array([0, 0, 0, 0, 1, 0]))
    tr.stats['channel'] = 'MRP'
    st.append(tr)

    tr = obspy.Trace(data=np.array([0, 0, 0, 0, 0, 1]))
    tr.stats['channel'] = 'MTP'
    st.append(tr)

    st_ref = obspy.Stream()
    tr = obspy.Trace(data=np.array([1, 2, 3, 4, 5, 6]))
    st_ref.append(tr)

    tensor_new = invert_MT(st_ref, st)

    print(tensor_new)

    npt.assert_allclose(1, tensor_new.m_rr,
                        rtol=1e-3, err_msg='MRR not the same')
    npt.assert_allclose(2, tensor_new.m_tt,
                        rtol=1e-3, err_msg='MTT not the same')
    npt.assert_allclose(3, tensor_new.m_pp,
                        rtol=1e-3, err_msg='MPP not the same')
    npt.assert_allclose(4, tensor_new.m_rt,
                        rtol=1e-3, err_msg='MRT not the same')
    npt.assert_allclose(5, tensor_new.m_rp,
                        rtol=1e-3, err_msg='MRP not the same')
    npt.assert_allclose(6, tensor_new.m_tp,
                        rtol=1e-3, err_msg='MTP not the same')


def test_create_Toeplitz():
    from stfinv.utils.inversion import _create_Toeplitz
    # even length
    d1 = np.array([1., 0., 0., 0., 1., 2., 1., 0., 0., 1])
    d2 = np.array([0., 0., 1., 3., 2., 1., 0., 0., 0., 0])

    G = _create_Toeplitz(d2)
    npt.assert_allclose(np.matmul(G, d1),
                        np.convolve(d1, d2, 'same'),
                        atol=1e-7, rtol=1e-7)

    # odd length
    d1 = np.array([1., 0., 0., 0., 1., 2., 1., 0., 0.])
    d2 = np.array([0., 0., 1., 3., 2., 1., 0., 0., 0.])

    G = _create_Toeplitz(d2)
    npt.assert_allclose(np.matmul(G, d1),
                        np.convolve(d1, d2, 'same'),
                        atol=1e-7, rtol=1e-7)


def test_create_Toeplitz_mult():
    from stfinv.utils.inversion import _create_Toeplitz_mult
    tr = obspy.Trace(data=np.array([0., 0., 0., 0., 1., 2., 1., 0., 0.]))
    st = obspy.Stream(tr)
    tr = obspy.Trace(data=np.array([0., 0., 1., 3., 2., 1., 0., 0., 0.]))
    st.append(tr)

    d = np.array([0., 0., 1., 1., 2., 1., 1., 0., 0.])
    G = _create_Toeplitz_mult(st)

    ref = [np.convolve(st[0].data, d, 'same'),
           np.convolve(st[1].data, d, 'same')]
    res = np.matmul(G, d).reshape(2, 9)
    npt.assert_allclose(ref, res, atol=1e-7, rtol=1e-7)


def test_invert_STF():
    from stfinv.utils.inversion import invert_STF
    tr = obspy.Trace(data=np.array([0., 0., 0., 0., 1., 2., 1., 0., 0.]))
    st_synth = obspy.Stream(tr)
    tr = obspy.Trace(data=np.array([0., 0., 1., 3., 2., 1., 0., 0., 0.]))
    st_synth.append(tr)

    stf_ref = np.array([0., 0., 1., 1., 0., 1., 1., 0., 0.])

    tr = obspy.Trace(data=np.convolve(st_synth[0].data, stf_ref, 'same'))
    st_data = obspy.Stream(tr)
    tr = obspy.Trace(data=np.convolve(st_synth[1].data, stf_ref, 'same'))
    st_data.append(tr)

    stf = invert_STF(st_data, st_synth)

    halflen = (len(stf) + 1) / 2
    stf = np.r_[stf[halflen:], stf[0:halflen]]

    npt.assert_allclose(stf, stf_ref, rtol=1e-7, atol=1e-10)


def test_invert_STF_dampened():
    from stfinv.utils.inversion import invert_STF
    tr = obspy.Trace(data=np.array([0., 0., 0., 0., 1., 2., 1., 0., 0.]))
    st_synth = obspy.Stream(tr)
    tr = obspy.Trace(data=np.array([0., 0., 1., 3., 2., 1., 0., 0., 0.]))
    st_synth.append(tr)

    stf_ref = np.array([0., 0., 1., 1., 0., 1., 1., 0., 0.])

    tr = obspy.Trace(data=np.convolve(st_synth[0].data, stf_ref, 'same'))
    st_data = obspy.Stream(tr)
    tr = obspy.Trace(data=np.convolve(st_synth[1].data, stf_ref, 'same'))
    st_data.append(tr)

    stf = invert_STF(st_data, st_synth, method='dampened', eps=1e-4)

    npt.assert_allclose(stf, stf_ref, rtol=1e-2, atol=1e-10)
