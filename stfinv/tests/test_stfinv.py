from matplotlib import use
use('Agg')
import numpy as np
import numpy.testing as npt
import obspy
import instaseis
import os
import stfinv
from obspy.geodetics import gps2dist_azimuth


# def test_inversion():
#     data_path = './stfinv/data/dis*.BHZ'
#     db_path='syngine://ak135f_2s'
#     # db_path = '/import/tethys-2g-nas-dump/instaseis/PREMiso_2s/'
#     for depth in range(5, 15):
#         stfinv.inversion(data_path=data_path,
#                          event_file='./stfinv/data/virginia.xml',
#                          #pre_offset=30,
#                          #post_offset=72.3,
#                          db_path=db_path,
#                          depth_in_m=depth * 1e3)


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

    st_synth = stfinv.calc_synthetic_from_grf6(st_grf6,
                                               st_ref,
                                               tensor=tensor,
                                               stf=[1])

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


def test_pick():

    signal = np.array([3, 3, 4, 4, 4, 3, 2, 1, 0, 3, 2,
                       1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
    tr = obspy.Trace(data=signal)
    threshold = 3.0
    idx = stfinv.pick(tr, threshold)
    npt.assert_equal(idx, 2, err_msg='Pick first surpassing of threshold')

    signal = np.array([4, 4, 4, 4, 4, 3, 2, 1, 0, 3, 2,
                       1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
    tr = obspy.Trace(data=signal)
    threshold = 3.0
    idx = stfinv.pick(tr, threshold)
    npt.assert_equal(idx, 16, err_msg='Pick first surpassing of threshold')

    signal = - np.array([4, 4, 4, 4, 4, 3, 2, 1, 0, 3, 2,
                         1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
    tr = obspy.Trace(data=signal)
    threshold = 3.0
    idx = stfinv.pick(tr, threshold)
    npt.assert_equal(idx, 16, err_msg='Negative signals')

    signal = np.array([4, 4, 4, 4, 4, 3, 2, 1, 0, 3, 2,
                       1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
    tr = obspy.Trace(data=signal)
    threshold = 9.0
    idx = stfinv.pick(tr, threshold)
    npt.assert_equal(idx, 0.0, err_msg='Pick zero, if all below threshold')


def test_pick_stream():
    testdata_dir = 'stfinv/data'
    st = obspy.read(os.path.join(testdata_dir,
                    'data_II.BFO.00.MPP.SAC'))
    arr_times = stfinv.pick_stream(st)

    npt.assert_allclose(arr_times['BFO.00'], 12.1)


def test_taper_signal():
    tr_ref = obspy.Trace(data=np.ones(100), header={'delta': 0.1})
    result_ref = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.00647486868104, 0.0257317790264, 0.0572719871734, 0.100278618298,
         0.153637823245, 0.215967626634, 0.285653719298, 0.360891268042,
         0.439731659872, 0.520132970055, 0.600012846888, 0.677302443521, 0.75,
         0.816222687798, 0.874255374086, 0.922595042772, 0.959989721829,
         0.985470908713, 0.998378654067, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999291347838,
         0.993634177361, 0.982383934407, 0.965668088726, 0.943676037528,
         0.916656959541, 0.884916991715, 0.848815760567, 0.808762307473,
         0.76521045406, 0.718653660229, 0.669619433059, 0.618663349936,
         0.566362763642, 0.513310260719, 0.460106947223, 0.407355637956,
         0.35565402633, 0.305587912263, 0.257724564834, 0.212606294893,
         0.170744310467, 0.132612924568, 0.0986441810345, 0.0692229593031,
         0.0446826135725, 0.0253011957658, 0.0112983050911, 0.00283259989493,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stfinv.taper_signal(tr_ref, t_begin=3, t_end=6)
    npt.assert_allclose(tr_ref.data, result_ref, atol=1e-10,
                        err_msg='Tapering equal to reference')

    tr_ref = obspy.Trace(data=np.ones(100), header={'delta': 0.1})
    result_ref = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.00647486868104, 0.0257317790264, 0.0572719871734, 0.100278618298,
         0.153637823245, 0.215967626634, 0.285653719298, 0.360891268042,
         0.439731659872, 0.520132970055, 0.600012846888, 0.677302443521, 0.75,
         0.816222687798, 0.874255374086, 0.922595042772, 0.959989721829,
         0.985470908713, 0.998378654067, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999825770934,
         0.998432666864, 0.995650341553, 0.991486549842, 0.985952896965,
         0.979064806205, 0.970841475907, 0.961305825967, 0.950484433951,
         0.938407461021])
    stfinv.taper_signal(tr_ref, t_begin=3, t_end=9)

    npt.assert_allclose(tr_ref.data, result_ref, atol=1e-10,
                        err_msg='Tapering equal to reference')


def test_taper_before_arrival():

    testdata_dir = 'stfinv/data'
    st_data = obspy.read(os.path.join(testdata_dir,
                                      'data_II.BFO.00.BHZ.SAC'))
    st_synth = obspy.read(os.path.join(testdata_dir,
                                       'data_II.BFO.00.MPP.SAC'))

    # code = '%s.%s' % (st_data[0].stats.station,
    #                   st_data[0].stats.location)

    # plt.plot(st_data[0].times(), st_data[0].data, label='untapered')
    len_win, arr_times = stfinv.taper_before_arrival(st_data,
                                                     st_synth)

    # plt.plot(st_data[0].times(), st_data[0].data, label='tapered')
    # plt.plot(st_synth[0].times(), st_synth[0].data * 1e18,
    #          label='synth')
    # plt.plot(arr_times[code] * np.ones(2), (-1e-6, 1e-6))
    # plt.legend()
    # plt.savefig('taper_before_arrival.png')
