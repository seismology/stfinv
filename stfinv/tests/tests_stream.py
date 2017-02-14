#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  tests_stream.py
#   Purpose:   Test routines for Stream class
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

from stfinv.utils.stream import read, Stream, taper_signal, get_grf6
import os
import numpy as np
import numpy.testing as npt
import obspy
import instaseis


def test_get_grf6():

    db = instaseis.open_db('syngine://prem_a_20s')

    reclat = 10.0
    reclon = 10.0

    cat = obspy.read_events('./stfinv/data/virginia.xml')
    # Define Moment tensor
    tensor = cat[0].focal_mechanisms[0].moment_tensor.tensor
    # tensor = obspy.core.event.Tensor(m_rr=1e20, m_pp=-2e20, m_tt=0.5e20,
    #                                  m_rt=5e19, m_rp=-7e19, m_tp=-1e20)

    # Get a reference trace with normal instaseis
    rec = instaseis.Receiver(latitude=reclat, longitude=reclon,
                             network='XX', station='YY', location='00')

    # src = instaseis.Source(latitude=evlat, longitude=evlon,
    #                        m_rr=tensor.m_rr,
    #                        m_tt=tensor.m_tt,
    #                        m_pp=tensor.m_pp,
    #                        m_tp=tensor.m_tp,
    #                        m_rt=tensor.m_rt,
    #                        m_rp=tensor.m_rp,
    #                        depth_in_m=evdepth)
    src = instaseis.Source.parse(cat[0])

    st_ref = db.get_seismograms(src, rec, dt=0.1, components='Z')

    # Get a synthetic
    st_grf6 = Stream()
    st_grf6 += get_grf6(db, origin=cat[0].origins[0],
                        rec_lat=reclat, rec_lon=reclon,
                        dt=0.1, stats=st_ref[0].stats,
                        depth_in_m=cat[0].origins[0].depth)

    st_synth = st_grf6.calc_synthetic_from_grf6(st_ref,
                                                tensor=tensor,
                                                stf=[1])

    # st_synth.plot(outfile='synth.png')
    # st_ref.plot(outfile='ref.png')

    npt.assert_allclose(st_ref[0].data, st_synth[0].data,
                        rtol=5e-1,
                        atol=1e-7,
                        err_msg='Synthetic not the same')


def test_pick():
    testdata_dir = 'stfinv/data'
    st = read(os.path.join(testdata_dir,
                           'data_II.BFO.00.MPP.SAC'))
    arr_times = st.pick()

    npt.assert_allclose(arr_times['BFO.00'], 12.1)


def test_filter_bad_waveforms():

    CC = dict()

    tr = obspy.Trace(header={'station': 'AAA',
                             'location': '00'})
    st = Stream(traces=tr)
    code = '%s.%s' % (tr.stats.station, tr.stats.location)
    CC[code] = 0.1

    tr = obspy.Trace(header={'station': 'BBB',
                             'location': '00'})
    st.append(tr)
    code = '%s.%s' % (tr.stats.station, tr.stats.location)
    CC[code] = 0.8

    tr = obspy.Trace(header={'station': 'CCC',
                             'location': '00'})
    st.append(tr)
    code = '%s.%s' % (tr.stats.station, tr.stats.location)
    CC[code] = -0.9

    tr = obspy.Trace(header={'station': 'DDD',
                             'location': '00'})
    st.append(tr)
    code = '%s.%s' % (tr.stats.station, tr.stats.location)
    CC[code] = 0.6

    st_filter = st.filter_bad_waveforms(CC=CC, CClim=0.6)

    npt.assert_equal(len(st_filter), 2)
    npt.assert_string_equal(str(st_filter[0].stats.station), 'BBB')
    npt.assert_string_equal(str(st_filter[1].stats.station), 'DDD')


def test_get_station_coordinates():

    st = read('./stfinv/data/data_wo_information.mseed')
    inv = obspy.read_inventory('./stfinv/data/some_stations.xml')

    st.get_station_coordinates()
    for tr in st:
        stats = tr.stats
        stat = inv.select(network=stats.network,
                          station=stats.station,
                          location=stats.location,
                          channel=stats.channel)[0][0]
        npt.assert_allclose(stat.longitude, stats.sac['stlo'], atol=1e-2)
        npt.assert_allclose(stat.latitude, stats.sac['stla'], atol=1e-2)


def test_get_synthetics():
    # Try to load 3 stations, out of which 2 are in range for P
    db = instaseis.open_db('syngine://prem_a_20s')
    cat = obspy.read_events('./stfinv/data/virginia.xml')
    st = read('./stfinv/data/dis.II.BFO.00.BHZ')
    st += read('./stfinv/data/dis.GE.DAG..BHZ')
    st += read('./stfinv/data/dis.G.CRZF.00.BHZ')
    st_data, st_syn = st.get_synthetics(db=db, origin=cat[0].origins[0],
                                        out_dir='/tmp')

    npt.assert_equal(len(st_data), 2)
    npt.assert_equal(len(st_syn), 12)

    for istat in range(0, 2):
        channels = ['MPP', 'MRP', 'MRR', 'MRT', 'MTP', 'MTT']
        for channel in channels:
            st_test = st_syn.select(station=st_data[istat].stats.station,
                                    network=st_data[istat].stats.network,
                                    location=st_data[istat].stats.location,
                                    channel=channel)
            npt.assert_equal(len(st_test), 1)

        for tr in st_syn[istat * 6:(istat + 1) * 6]:
            npt.assert_string_equal(str(tr.stats.station),
                                    str(st_data[istat].stats.station))
            npt.assert_string_equal(str(tr.stats.location),
                                    str(st_data[istat].stats.location))
            npt.assert_string_equal(str(tr.stats.network),
                                    str(st_data[istat].stats.network))

            npt.assert_equal(tr.stats.npts, st_data[istat].stats.npts)
            npt.assert_allclose(tr.stats.delta, st_data[istat].stats.delta)
            npt.assert_allclose(float(tr.stats.starttime),
                                float(st_data[istat].stats.starttime))


def test_shift_waveform():

    data_test = [0, 0, 1, 2, 1, 0, 0]
    tr_test = obspy.Trace(np.array(data_test),
                          header=dict(station='AAA',
                                      location='00',
                                      delta=0.1))
    st_ref = Stream(tr_test)
    tr_test = obspy.Trace(np.array(data_test),
                          header=dict(station='BBB',
                                      location='00',
                                      delta=0.1))
    st_ref.append(tr_test)

    dt = dict()
    dt['AAA.00'] = 0.2
    dt['BBB.00'] = -0.1
    st_shift = st_ref.copy()
    st_shift.shift_waveform(dt)

    # Shift backwards by 0.2 s
    data_ref = [0, 0, 0, 0, 1, 2, 1]
    npt.assert_allclose(st_shift.select(station='AAA')[0].data,
                        data_ref,
                        rtol=1e-5, atol=1e-3,
                        err_msg='Shifted data not as expected')

    # Shift forwards by 0.1 s
    data_ref = [0, 1, 2, 1, 0, 0, 0]
    npt.assert_allclose(st_shift.select(station='BBB')[0].data,
                        data_ref, rtol=1e-5, atol=1e-3,
                        err_msg='Shifted data not as expected')


def test_calc_timeshift():

    data_test = [0, 0, 1, 2, 1, 0, 0]
    tr_test = obspy.Trace(np.array(data_test),
                          header=dict(station='AAA',
                                      location='00',
                                      delta=0.1))
    st_ref = Stream(tr_test)
    tr_test = obspy.Trace(np.array(data_test),
                          header=dict(station='BBB',
                                      location='00',
                                      delta=0.1))
    st_ref.append(tr_test)

    # Shift backwards by 0.2 s
    dt = dict()
    dt['AAA.00'] = 0.2
    dt['BBB.00'] = -0.1
    st_shift = st_ref.copy()
    st_shift.shift_waveform(dt)
    dt_res, CC = st_shift.calc_timeshift(st_ref)

    npt.assert_allclose(dt_res['AAA.00'], dt['AAA.00'], rtol=1e-5)
    npt.assert_allclose(dt_res['BBB.00'], dt['BBB.00'], rtol=1e-5)

    # Shift backwards by 0.2 s, reverse order of traces in stream
    dt = dict()
    dt['AAA.00'] = 0.2
    dt['BBB.00'] = -0.1
    st_shift = st_ref.copy()
    st_shift.shift_waveform(dt)
    st_shift.sort(keys=['station'], reverse=True)
    dt_res, CC = st_shift.calc_timeshift(st_ref)

    npt.assert_allclose(dt_res['AAA.00'], dt['AAA.00'], rtol=1e-5)
    npt.assert_allclose(dt_res['BBB.00'], dt['BBB.00'], rtol=1e-5)


def test_calc_amplitude_misfit():

    data_test = [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]
    tr_test = obspy.Trace(np.array(data_test),
                          header=dict(station='AAA',
                                      location='00',
                                      delta=0.1))
    st_ref = Stream(tr_test)
    tr_test = obspy.Trace(np.array(data_test),
                          header=dict(station='BBB',
                                      location='00',
                                      delta=0.1))
    st_ref.append(tr_test)

    st_mult = st_ref.copy()

    st_mult.select(station='AAA')[0].data *= 2
    st_mult.select(station='BBB')[0].data *= 0.5

    dA = st_mult.calc_amplitude_misfit(st_ref)

    npt.assert_almost_equal(dA['AAA.00'], 2.0, decimal=5)
    npt.assert_almost_equal(dA['BBB.00'], 0.5, decimal=5)

    # Check when order is mixed up
    st_mult = st_ref.copy()

    st_mult.select(station='AAA')[0].data *= 2
    st_mult.select(station='BBB')[0].data *= 0.5

    st_mult.sort(keys=['station'], reverse=True)

    dA = st_mult.calc_amplitude_misfit(st_ref)

    npt.assert_almost_equal(dA['AAA.00'], 2.0, decimal=5)
    npt.assert_almost_equal(dA['BBB.00'], 0.5, decimal=5)


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
    taper_signal(tr_ref, t_begin=3, t_end=6)
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
    taper_signal(tr_ref, t_begin=3, t_end=9)

    npt.assert_allclose(tr_ref.data, result_ref, atol=1e-10,
                        err_msg='Tapering equal to reference')


def test_taper_before_arrival():

    testdata_dir = 'stfinv/data'
    st_data = Stream()
    st_data += obspy.read(os.path.join(testdata_dir,
                                       'data_II.BFO.00.BHZ.SAC'))
    st_synth = Stream()
    st_synth += obspy.read(os.path.join(testdata_dir,
                                        'data_II.BFO.00.MPP.SAC'))

    # code = '%s.%s' % (st_data[0].stats.station,
    #                   st_data[0].stats.location)

    # plt.plot(st_data[0].times(), st_data[0].data, label='untapered')
    len_win, arr_times = st_data.taper_before_arrival(st_synth)

    # plt.plot(st_data[0].times(), st_data[0].data, label='tapered')
    # plt.plot(st_synth[0].times(), st_synth[0].data * 1e18,
    #          label='synth')
    # plt.plot(arr_times[code] * np.ones(2), (-1e-6, 1e-6))
    # plt.legend()
    # plt.savefig('taper_before_arrival.png')


# def test_calc_L2_misfit():
#
#     data_test = [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]
#     tr_test = obspy.Trace(np.array(data_test),
#                           header=dict(station='AAA',
#                                       location='00',
#                                       delta=0.1))
#     st_ref = Stream(tr_test)
#     tr_test = obspy.Trace(np.array(data_test),
#                           header=dict(station='BBB',
#                                       location='00',
#                                       delta=0.1))
#     st_ref.append(tr_test)
#
#     st_pert = st_ref.copy()
#
#     st_pert.select(station='AAA')[0].data[3] = 0
#     st_pert.select(station='BBB')[0].data[3] = -1
#
#     RMS = st_pert.calc_L2_misfit(st_ref)
#
#     npt.assert_almost_equal(RMS, np.sqrt(2) + np.sqrt(3), decimal=5)

# def test_pick():
#
#     signal = np.array([3, 3, 4, 4, 4, 3, 2, 1, 0, 3, 2,
#                        1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
#     tr = obspy.Trace(data=signal)
#     threshold = 3.0
#     idx = stfinv.pick(tr, threshold)
#     npt.assert_equal(idx, 2, err_msg='Pick first surpassing of threshold')
#
#     signal = np.array([4, 4, 4, 4, 4, 3, 2, 1, 0, 3, 2,
#                        1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
#     tr = obspy.Trace(data=signal)
#     threshold = 3.0
#     idx = stfinv.pick(tr, threshold)
#     npt.assert_equal(idx, 16, err_msg='Pick first surpassing of threshold')
#
#     signal = - np.array([4, 4, 4, 4, 4, 3, 2, 1, 0, 3, 2,
#                          1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
#     tr = obspy.Trace(data=signal)
#     threshold = 3.0
#     idx = stfinv.pick(tr, threshold)
#     npt.assert_equal(idx, 16, err_msg='Negative signals')
#
#     signal = np.array([4, 4, 4, 4, 4, 3, 2, 1, 0, 3, 2,
#                        1, 0, 0, 1, 1, 4, 8, 7, 6, 5, 0])
#     tr = obspy.Trace(data=signal)
#     threshold = 9.0
#     idx = stfinv.pick(tr, threshold)
#     npt.assert_equal(idx, 0.0, err_msg='Pick zero, if all below threshold')
#
