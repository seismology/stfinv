import numpy as np
import numpy.testing as npt
import instaseis
from obspy.geodetics import gps2dist_azimuth
import obspy
import matplotlib
matplotlib.use('Agg')
import stfinv


# def test_inversion():
#     data_path = 'stfinv/data/dis*.BHZ'
#     stfinv.inversion(data_path=data_path,
#                      event_file='stfinv/data/virginia.xml',
#                      db_path='syngine://ak135f_2s')
#
#
def test_seiscomp_to_moment_tensor():
    # import matplotlib.pyplot as plt

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
    km2deg = 360.0 / (2 * np.pi * 6378137.0)
    distance *= km2deg

    gf_synth = db.get_greens_function(epicentral_distance_in_degree=distance,
                                      source_depth_in_m=evdepth,
                                      dt=0.1)

    st_grf6 = stfinv.seiscomp_to_moment_tensor(gf_synth,
                                               azimuth=azi,
                                               scalmom=1)

    rec = instaseis.Receiver(latitude=reclat, longitude=reclon)

    # Mxx/Mtt
    src_mtt = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_tt=1e20, depth_in_m=evdepth)
    m_tt_ref = db.get_seismograms(src_mtt, rec, dt=0.1, components='Z')[0].data

    m_tt = st_grf6.select(channel='MTT')[0].data * 1e20
    npt.assert_allclose(m_tt, m_tt_ref, atol=1e-6,
                        err_msg='M_tt not the same')
    # plt.plot(m_tt, label='GRF6')
    # plt.plot(m_tt_ref, label='REF')
    # plt.savefig('MTT.png')
    # plt.close()

    # Myy/Mpp
    src_mpp = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_pp=1e20, depth_in_m=evdepth)
    m_pp_ref = db.get_seismograms(src_mpp, rec, dt=0.1, components='Z')[0].data

    m_pp = st_grf6.select(channel='MPP')[0].data * 1e20
    npt.assert_allclose(m_pp, m_pp_ref, atol=1e-6,
                        err_msg='M_pp not the same')
    # plt.plot(m_pp, label='GRF6')
    # plt.plot(m_pp_ref, label='REF')
    # plt.savefig('MPP.png')
    # plt.close()

    # Mzz/Mrr
    src_mrr = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_rr=1e20, depth_in_m=evdepth)
    m_rr_ref = db.get_seismograms(src_mrr, rec, dt=0.1, components='Z')[0].data

    m_rr = st_grf6.select(channel='MRR')[0].data * 1e20
    npt.assert_allclose(m_rr, m_rr_ref, atol=1e-6,
                        err_msg='M_rr not the same')
    # plt.plot(m_rr, label='GRF6')
    # plt.plot(m_rr_ref, label='REF')
    # plt.savefig('MRR.png')
    # plt.close()

    # Mxy/Mtp
    src_mtp = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_tp=1e20, depth_in_m=evdepth)
    m_tp_ref = db.get_seismograms(src_mtp, rec, dt=0.1, components='Z')[0].data

    m_tp = st_grf6.select(channel='MTP')[0].data * 1e20
    npt.assert_allclose(m_tp, m_tp_ref, atol=1e-6,
                        err_msg='M_tp not the same')
    # plt.plot(m_tp, label='GRF6')
    # plt.plot(m_tp_ref, label='REF')
    # plt.savefig('MTP.png')
    # plt.close()

    # Mxz/Mtr
    src_mrt = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_rt=1e20, depth_in_m=evdepth)
    m_rt_ref = db.get_seismograms(src_mrt, rec, dt=0.1, components='Z')[0].data

    m_rt = st_grf6.select(channel='MRT')[0].data * 1e20
    npt.assert_allclose(m_rt, m_rt_ref, atol=1e-6,
                        err_msg='M_rt not the same')
    # plt.plot(m_rt, label='GRF6')
    # plt.plot(m_rt_ref, label='REF')
    # plt.savefig('MRT.png')
    # plt.close()

    # Myz/Mpr
    src_mrp = instaseis.Source(latitude=evlat, longitude=evlon,
                               m_rp=1e20, depth_in_m=evdepth)
    m_rp_ref = db.get_seismograms(src_mrp, rec, dt=0.1, components='Z')[0].data

    m_rp = st_grf6.select(channel='MRP')[0].data * 1e20
    npt.assert_allclose(m_rp, m_rp_ref, atol=1e-6,
                        err_msg='M_rp not the same')
    # plt.plot(m_rp, label='GRF6')
    # plt.plot(m_rp_ref, label='REF')
    # plt.savefig('MRP.png')
    # plt.close()


def test_calc_synthetic_from_grf6():

    db = instaseis.open_db('syngine://prem_a_20s')

    evlat = 50.0
    evlon = 8.0

    reclat = 10.0
    reclon = 10.0

    evdepth = 10.0

    # gps2dist_azimuth is not optimal here, since we need the geographical
    # coordinates, and not WGS84. Try to  around it by switching ellipticity
    # off.
    distance, azi, bazi = gps2dist_azimuth(evlat, evlon,
                                           reclat, reclon,
                                           f=0.0)

    # The following functions expect distance in degrees
    km2deg = 360.0 / (2 * np.pi * 6378137.0)
    distance *= km2deg

    # Define Moment tensor
    tensor = obspy.core.event.Tensor(m_rr=0e20, m_pp=0e20, m_tt=0e20,
                                     m_rt=0.0, m_rp=0.0, m_tp=-1e20)

    # Get a reference trace with normal instaseis
    rec = instaseis.Receiver(latitude=reclat, longitude=reclon)

    src = instaseis.Source(latitude=evlat, longitude=evlon,
                           m_rr=tensor.m_rr,
                           m_tt=tensor.m_tt,
                           m_pp=tensor.m_pp,
                           m_tp=tensor.m_tp,
                           m_rt=tensor.m_rt,
                           m_rp=tensor.m_rp,
                           depth_in_m=evdepth)

    st_ref = db.get_seismograms(src, rec, dt=0.1, components='Z')

    # get Greens functions
    gf_synth = db.get_greens_function(epicentral_distance_in_degree=distance,
                                      source_depth_in_m=evdepth,
                                      dt=0.1)

    # Convert to GRF6 format
    st_grf6 = stfinv.seiscomp_to_moment_tensor(gf_synth,
                                               azimuth=azi,
                                               scalmom=1,
                                               stats=st_ref[0].stats)

    st_synth = stfinv.calc_synthetic_from_grf6(st_grf6, st_ref, tensor)

    # st_synth.plot(outfile='synth.png')
    # st_ref.plot(outfile='ref.png')

    npt.assert_allclose(st_ref[0].data, st_synth[0].data,
                        atol=1e-6,
                        err_msg='Synthetic not the same')


def test_shift_waveform():

    data_test = [0, 0, 1, 2, 1, 0, 0]
    tr_test = obspy.Trace(np.array(data_test))
    tr_test.stats.delta = 0.1

    # Shift backwards by 0.2 s
    tr_shift = stfinv.shift_waveform(tr_test, 0.2)
    data_ref = [0, 0, 0, 0, 1, 2, 1]

    npt.assert_allclose(tr_shift.data, data_ref, rtol=1e-5, atol=1e-3,
                        err_msg='Shifted data not as expected')
    # Shift forwards by 0.2 s
    tr_shift = stfinv.shift_waveform(tr_test, -0.2)
    data_ref = [1, 2, 1, 0, 0, 0, 0]

    npt.assert_allclose(tr_shift.data, data_ref, rtol=1e-5, atol=1e-3,
                        err_msg='Shifted data not as expected')


def test_calc_timeshift():

    data_test = [0, 0, 1, 2, 1, 0, 0]
    tr_test = obspy.Trace(np.array(data_test))
    tr_test.stats.delta = 0.1
    tr_test.stats['station'] = 'REF'
    tr_test.stats['location'] = '00'
    st_test = obspy.Stream(traces=tr_test)

    # Shift backwards by 0.2 s
    tr_shift = stfinv.shift_waveform(tr_test, 0.1)
    st_shift = obspy.Stream(traces=tr_shift)
    dt, CC = stfinv.calc_timeshift(st_test, st_shift)

    npt.assert_allclose(dt['REF.00'], -0.1, rtol=1e-5)


def test_calc_amplitude_misfit():

    data_test = [0, 0, 1, 2, 1, 0, 0]
    tr_test = obspy.Trace(np.array(data_test))
    tr_test.stats.delta = 0.1
    tr_test.stats['station'] = 'REF'
    tr_test.stats['location'] = '00'
    st_test = obspy.Stream(traces=tr_test)

    # Multiply by 2.
    tr_mult = obspy.Trace(np.array(data_test) * 2.0)
    tr_mult.stats.delta = 0.1
    tr_mult.stats['station'] = 'REF'
    tr_mult.stats['location'] = '00'
    st_mult = obspy.Stream(traces=tr_mult)
    dA = stfinv.calc_amplitude_misfit(st_test, st_mult)

    npt.assert_almost_equal(dA['REF.00'], 0.5, decimal=5)


def test_get_station_coordinates():
    st = obspy.read('./stfinv/data/data_wo_information.mseed')
    inv = obspy.read_inventory('./stfinv/data/some_stations.xml')

    stfinv.get_station_coordinates(st)
    for tr in st:
        stats = tr.stats
        stat = inv.select(network=stats.network,
                          station=stats.station,
                          location=stats.location,
                          channel=stats.channel)[0][0]
        npt.assert_equal(stat.longitude, stats.sac['stlo'])
        npt.assert_equal(stat.latitude, stats.sac['stla'])


def test_create_Toeplitz():
    print('even length')
    d1 = np.array([1., 0., 0., 0., 1., 2., 1., 0., 0., 1])
    d2 = np.array([0., 0., 1., 3., 2., 1., 0., 0., 0., 0])

    G = stfinv.create_Toeplitz(d2)
    npt.assert_allclose(np.matmul(G, d1),
                        np.convolve(d1, d2, 'same'),
                        atol=1e-7, rtol=1e-7)

    print('odd length')
    d1 = np.array([1., 0., 0., 0., 1., 2., 1., 0., 0.])
    d2 = np.array([0., 0., 1., 3., 2., 1., 0., 0., 0.])

    G = stfinv.create_Toeplitz(d2)
    npt.assert_allclose(np.matmul(G, d1),
                        np.convolve(d1, d2, 'same'),
                        atol=1e-7, rtol=1e-7)


def test_create_Toeplitz_mult():
    tr = obspy.Trace(data=np.array([0., 0., 0., 0., 1., 2., 1., 0., 0.]))
    st = obspy.Stream(tr)
    tr = obspy.Trace(data=np.array([0., 0., 1., 3., 2., 1., 0., 0., 0.]))
    st.append(tr)

    d = np.array([0., 0., 1., 1., 2., 1., 1., 0., 0.])
    G = stfinv.create_Toeplitz_mult(st)

    ref = [np.convolve(st[0].data, d, 'same'),
           np.convolve(st[1].data, d, 'same')]
    res = np.matmul(G, d).reshape(2, 9)
    npt.assert_allclose(ref, res, atol=1e-7, rtol=1e-7)


def test_invert_STF():
    tr = obspy.Trace(data=np.array([0., 0., 0., 0., 1., 2., 1., 0., 0.]))
    st_synth = obspy.Stream(tr)
    tr = obspy.Trace(data=np.array([0., 0., 1., 3., 2., 1., 0., 0., 0.]))
    st_synth.append(tr)

    stf_ref = np.array([0., 0., 1., 1., -2., 1., 1., 0., 0.])

    tr = obspy.Trace(data=np.convolve(st_synth[0].data, stf_ref, 'same'))
    st_data = obspy.Stream(tr)
    tr = obspy.Trace(data=np.convolve(st_synth[1].data, stf_ref, 'same'))
    st_data.append(tr)

    stf = stfinv.invert_STF(st_data, st_synth)

    npt.assert_allclose(stf, stf_ref, rtol=1e-7, atol=1e-10)


def test_pick():

    signal = np.array([3, 3, 4, 4, 4, 3, 2, 1, 0, 3, 2, \
                       1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
    threshold=3
    idx = stfinv.pick(signal, threshold)
    npt.assert_equal(idx, 2, err_msg='Pick first surpassing of threshold')

    signal = np.array([4, 4, 4, 4, 4, 3, 2, 1, 0, 3, 2, \
                       1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
    threshold=3
    idx = stfinv.pick(signal, threshold)
    npt.assert_equal(idx, 16, err_msg='Pick first surpassing of threshold')

