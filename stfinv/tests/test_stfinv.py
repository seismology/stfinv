import numpy as np
import numpy.testing as npt
import instaseis
from obspy.geodetics import gps2dist_azimuth
import obspy
from stfinv import seiscomp_to_moment_tensor, shift_waveform, calc_timeshift


def test_seiscomp_to_moment_tensor():

    db = instaseis.open_db('syngine://prem_a_20s')

    evlat = 50.0
    evlon = 8.0

    reclat = 10.0
    reclon = 10.0

    evdepth = 10.0

    # gps2dist_azimuth is not optimal here, since we need the geographical
    # coordinates, and not WGS84. Try to get around it by switching ellipticity
    # off.
    distance, azi, bazi = gps2dist_azimuth(evlat, evlon,
                                           reclat, reclon,
                                           f=0.0)

    # The following functions expect distance in degrees
    km2deg = 360.0 / (2*np.pi*6378137.0)
    distance *= km2deg

    gf_synth = db.get_greens_function(epicentral_distance_in_degree=distance,
                                      source_depth_in_m=evdepth,
                                      dt=0.1)

    rec = instaseis.Receiver(latitude=reclat, longitude=reclon)

    # Mxx/Mtt
    src_mxx = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_tt=1e20, depth_in_m=evdepth)
    m_xx_ref = db.get_seismograms(src_mxx, rec, dt=0.1, components='Z')[0].data

    # Myy/Mpp
    src_myy = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_pp=1e20, depth_in_m=evdepth)
    m_yy_ref = db.get_seismograms(src_myy, rec, dt=0.1, components='Z')[0].data

    # Mzz/Mrr
    src_mzz = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_rr=1e20, depth_in_m=evdepth)
    m_zz_ref = db.get_seismograms(src_mzz, rec, dt=0.1, components='Z')[0].data

    # Mxy/Mtp
    src_mxy = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_tp=1e20, depth_in_m=evdepth)
    m_xy_ref = db.get_seismograms(src_mxy, rec, dt=0.1, components='Z')[0].data

    # Mxz/Mtr
    src_mxz = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_rt=1e20, depth_in_m=evdepth)
    m_xz_ref = db.get_seismograms(src_mxz, rec, dt=0.1, components='Z')[0].data

    # Myz/Mpr
    src_myz = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_rp=1e20, depth_in_m=evdepth)
    m_yz_ref = db.get_seismograms(src_myz, rec, dt=0.1, components='Z')[0].data

    m_xx, m_yy, m_zz, m_xy, m_xz, m_yz = seiscomp_to_moment_tensor(gf_synth,
                                                                   azimuth=azi,
                                                                   scalmom=1e20)

    npt.assert_array_almost_equal(m_xx, m_xx_ref, decimal=3,
                                  err_msg='M_xx not the same')
    npt.assert_array_almost_equal(m_yy, m_yy_ref, decimal=3,
                                  err_msg='M_yy not the same')
    npt.assert_array_almost_equal(m_zz, m_zz_ref, decimal=3,
                                  err_msg='M_zz not the same')
    npt.assert_array_almost_equal(m_xy, m_xy_ref, decimal=3,
                                  err_msg='M_xy not the same')
    npt.assert_array_almost_equal(m_xz, -m_xz_ref, decimal=3,
                                  err_msg='M_xz not the same')
    npt.assert_array_almost_equal(m_yz, m_yz_ref, decimal=3,
                                  err_msg='M_yz not the same')


def test_shift_waveform():

    data_test = [0, 0, 1, 2, 1, 0, 0]
    tr_test = obspy.Trace(np.array(data_test))
    tr_test.stats.delta = 0.1

    # Shift backwards by 0.2 s
    tr_shift = shift_waveform(tr_test, 0.2)
    data_ref = [0, 0, 0, 0, 1, 2, 1]

    npt.assert_array_almost_equal(tr_shift.data, data_ref, decimal=3,
                                  err_msg='Shifted data not as expected')
    # Shift forwards by 0.2 s
    tr_shift = shift_waveform(tr_test, -0.2)
    data_ref = [1, 2, 1, 0, 0, 0, 0]

    npt.assert_array_almost_equal(tr_shift.data, data_ref, decimal=3,
                                  err_msg='Shifted data not as expected')

def test_calc_timeshift():

    data_test = [0, 0, 1, 2, 1, 0, 0]
    tr_test = obspy.Trace(np.array(data_test))
    tr_test.stats.delta = 0.1
    tr_test.stats['station'] = 'REF'
    tr_test.stats['location'] = '00'
    st_test = obspy.Stream(traces=tr_test)

    # Shift backwards by 0.2 s
    tr_shift = shift_waveform(tr_test, 0.1)
    st_shift = obspy.Stream(traces=tr_shift)
    dt = calc_timeshift(st_test, st_shift)

    npt.assert_almost_equal(dt['REF.00'], -0.1, decimal=5)
