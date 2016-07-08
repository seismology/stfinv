#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  tests_stream.py
#   Purpose:   Test routines for Stream class
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

from stfinv.utils.stream import Stream
from stfinv import correct_waveforms
import numpy.testing as npt
import numpy as np
import obspy


def test_correct_waveforms():

    data_test = [0, 0, 1, 2, 1, 0, 0]
    tr_test = obspy.Trace(np.array(data_test, dtype=float),
                          header=dict(station='AAA',
                                      location='00',
                                      delta=0.1))
    st_data = Stream(tr_test)

    data_test = [0, 0, 1, 0, 2, 1, 0]
    tr_test = obspy.Trace(np.array(data_test, dtype=float),
                          header=dict(station='BBB',
                                      location='00',
                                      delta=0.1))
    st_data.append(tr_test)

    data_test = [0, 0, 2, 0, 0, 1, 0]
    tr_test = obspy.Trace(np.array(data_test, dtype=float),
                          header=dict(station='CCC',
                                      location='00',
                                      delta=0.1))
    st_data.append(tr_test)

    st_synth = st_data.copy()
    st_grf6 = st_data.copy()
    dt_in = dict()
    dA_in = dict()

    # Perturb traces in st_data.
    # 1st trace is shifted by +0.1s and multiplied by 2
    dt_in['AAA.00'] = 0.1
    dA_in['AAA.00'] = 2.0
    st_data.select(station='AAA')[0].data *= dA_in['AAA.00']

    # 2nd trace is shifted by -0.1s and multiplied by 10
    dt_in['BBB.00'] = -0.1
    dA_in['BBB.00'] = 10.0
    st_data.select(station='BBB')[0].data *= dA_in['BBB.00']

    # 3rd trace is not shifted and multiplied by 0.25
    dt_in['CCC.00'] = 0.0
    dA_in['CCC.00'] = 0.25
    st_data.select(station='CCC')[0].data *= dA_in['CCC.00']

    st_data.shift_waveform(dt_in)

    st_data_corr, st_synth_corr, st_grf6_corr, \
        CC, dt_out, dA_out = \
        correct_waveforms(st_data,
                          st_synth,
                          st_grf6,
                          allow_negative_CC=False)  # (it == 0))

    # Check that st_data_corr is identical to st_data
    npt.assert_equal(st_data_corr, st_data)

    # Check the dt values
    npt.assert_equal(dt_in, dt_out)

    # Check the dA values
    npt.assert_equal(dA_in, dA_out)

    # Check that the waveforms are actually corrected, synthetic data
    npt.assert_allclose(st_data.select(station='AAA')[0].data,
                        st_synth_corr.select(station='AAA')[0].data,
                        atol=1e-10)
    npt.assert_allclose(st_data.select(station='BBB')[0].data,
                        st_synth_corr.select(station='BBB')[0].data,
                        atol=1e-10)
    npt.assert_allclose(st_data.select(station='CCC')[0].data,
                        st_synth_corr.select(station='CCC')[0].data,
                        atol=1e-10)

    # Check that the waveforms are actually corrected, grf6 data
    npt.assert_allclose(st_data.select(station='AAA')[0].data,
                        st_grf6_corr.select(station='AAA')[0].data,
                        atol=1e-10)
    npt.assert_allclose(st_data.select(station='BBB')[0].data,
                        st_grf6_corr.select(station='BBB')[0].data,
                        atol=1e-10)
    npt.assert_allclose(st_data.select(station='CCC')[0].data,
                        st_grf6_corr.select(station='CCC')[0].data,
                        atol=1e-10)
