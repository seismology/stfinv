#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  trace.py
#   Purpose:   Provide extended Trace class for stfinv
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

import obspy
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from scipy import signal
import scipy.fftpack as fft
import numpy as np
import os
import glob


class Stream(obspy.Stream):

    def __init__(self, traces=None):
        self.traces = []
        if isinstance(traces, obspy.Trace):
            traces = [traces]
        if traces:
            self.traces.extend(traces)

    def pick(self):
        """
        arr_times = pick_stream(st)

        Return arrival times in stream as a dictionary. Arrival is defined as
        crossing of threshold of 1 percent of maximum value of waveform.

        Parameters
        ----------
        st : obspy.Stream()
            Stream with signals.

        threshold : float
            Threshold


        Returns
        -------
        arr_times : dict
            Dictionary with arrival times.

        """
        arr_times = dict()

        for tr in self:
            threshold = max(abs(tr.data)) * 1e-2
            arr_time = pick(tr, threshold=threshold)
            arr_times['%s.%s' % (tr.stats.station,
                                 tr.stats.location)] = arr_time

        return arr_times

    def filter_bad_waveforms(self, CC, CClim):
        st_filtered = Stream()
        for tr in self:
            code = '%s.%s' % (tr.stats.station, tr.stats.location)
            if CC[code] >= CClim:
                st_filtered.append(tr)
        return st_filtered

    def get_station_coordinates(self, client_base_url='IRIS'):
        """
        get_station_coordinates(self, client_base_url='IRIS')

        Use an obspy FDSN client to retrieve station locations for all traces
        in the stream. The results are written into a sac header-like
        dictionary.

        Parameters
        ----------
        self : stfinv.Stream()
            Stream with signals.

        client_base_url : str
            String with valid URL for FDSN client.


        """
        client = Client(client_base_url)
        bulk = []
        for tr in self:
            stats = tr.stats

            # Correct Instaseis Streams
            if (stats.location == ''):
                stats.location = u'00'
                stats.channel = u'BH%s' % (stats.channel[2])

            bulk.append([stats.network,
                        stats.station,
                        stats.location,
                        stats.channel,
                        stats.starttime,
                        stats.endtime])
            inv = client.get_stations_bulk(bulk)

        for tr in self:
            stats = tr.stats
            if not hasattr(stats, 'sac'):
                stat = inv.select(network=stats.network,
                                  station=stats.station,
                                  location=stats.location,
                                  channel=stats.channel)
                if (len(stat) > 0):
                    stats.sac = dict(stla=stat[0][0].latitude,
                                     stlo=stat[0][0].longitude)
                else:
                    print('Could not find station %s.%s.%s.%s' %
                          (stats.network, stats.station, stats.location,
                           stats.channel))

    def seiscomp_to_moment_tensor(self, azimuth, stats=None, scalmom=1.0):
        """
        seiscomp_to_moment_tensor(st_in, azimuth, scalmom=1.0)

        Convert Green's functions from the SeisComp convention to the format
        of one Green's function per Moment Tensor component.

        Parameters
        ----------
        st_in : obspy.Stream
            Stream as returned by instaseis.database.greens_function()

        azimuth : float
            Azimuth of the station as seen from the earthquake in degrees.

        scalmom : float, optional
            Scalar moment of the earthquake. Default is one.

        Returns
        -------
        m_xx, m_yy, m_zz, m_xy, m_xz, m_yz : array
            Moment tensor-component-specific Green's functions for a station
            at the given azimuth.

        """

        azimuth *= np.pi / 180.
        m_tt = (self.select(channel='ZSS')[0].data *
                np.cos(2 * azimuth) / 2.) +           \
            self.select(channel='ZEP')[0].data / 3. - \
            self.select(channel='ZDD')[0].data / 6.
        m_tt *= scalmom

        m_pp = - (self.select(channel='ZSS')[0].data *
                  np.cos(2 * azimuth) / 2.) +          \
            self.select(channel='ZEP')[0].data / 3. - \
            self.select(channel='ZDD')[0].data / 6.
        m_pp *= scalmom

        m_rr = self.select(channel='ZEP')[0].data / 3. + \
            self.select(channel='ZDD')[0].data / 3.
        m_rr *= scalmom

        m_tp = self.select(channel='ZSS')[0].data * np.sin(2 * azimuth)
        m_tp *= scalmom

        m_rt = - self.select(channel='ZDS')[0].data * np.cos(azimuth)
        m_rt *= scalmom

        m_rp = self.select(channel='ZDS')[0].data * np.sin(azimuth)
        m_rp *= scalmom

        data = [m_tt, m_pp, m_rr, -m_tp, m_rt, m_rp]
        channels = ['MTT', 'MPP', 'MRR', 'MTP', 'MRT', 'MRP']

        st_grf6 = Stream()
        for icomp in range(0, 6):
            if stats:
                tr_new = obspy.Trace(data=data[icomp],
                                     header=stats)
            else:
                tr_new = obspy.Trace(data=data[icomp])
            tr_new.stats['channel'] = channels[icomp]
            st_grf6.append(tr_new)

        return st_grf6

    def get_synthetics(self, origin, db, out_dir='inversion',
                       depth_in_m=-1,
                       pre_offset=5.6, post_offset=20.0,
                       dist_min=30.0, dist_max=85.0, phase_list='P'):
        """
        get_synthetics(self, origin, db, out_dir='inversion',
                       depth_in_m=-1,
                       pre_offset=5.6, post_offset=20.0,
                       dist_min=30.0, dist_max=85.0, phase_list='P')

        Convert Green's functions from the SeisComp convention to the format
        of one Green's function per Moment Tensor component.

        Parameters
        ----------
        origin : obspy.core.event.origin
            Origin object that is used to calculate the synthetic seismograms.

        db : Instaseis.database
            Database from which to calculate the synthetic seismograms.

        out_dir : str
            Directory in which to write SAC files with the seismograms windows.

        depth_in_m : float
            Event depth to use. -1 means: use the depth from origin

        pre_offset : float
            Begin of time windows relative to phase arrival in seconds

        post_offset : float
            End of time windows relative to phase arrival in seconds

        dist_min : float
            Minimal distance of station-earthquake to use (in degree)

        dist_max : float
            Maximum distance of station-earthquake to use (in degree)

        phase_list : Tuple
            Phases for which to create seismogram snips.

        Returns
        -------
        st_data : stfinv.utils.Stream
            Stream with data snips

        st_synth_grf6 : stfinv.utils.Stream
            Stream with GRF6 synthetics

        """

        # Use origin depth as default
        if depth_in_m == -1:
            depth_in_m = origin.depth

        km2deg = 360.0 / (2 * np.pi * 6378137.0)

        model = TauPyModel(model="iasp91")

        # Check for existing Green's functions
        data_dir = os.path.join(out_dir, 'data')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        grf6_dir = os.path.join(out_dir, 'grf6')
        if not os.path.exists(grf6_dir):
            os.mkdir(grf6_dir)

        st_data, st_synth_grf6 = load_cut_files(data_directory=data_dir,
                                                grf6_directory=grf6_dir,
                                                depth_in_m=depth_in_m)

        npts_target = int((post_offset + pre_offset) / db.info.dt) + 1
        print('NPTS_TARGET : %d ' % npts_target)
        st_data = Stream()
        st_synth = Stream()

        for tr in self:
            tr_work = tr.copy()
            tr_work.resample(1. / db.info.dt)
            distance, azi, bazi = gps2dist_azimuth(tr.stats.sac['stla'],
                                                   tr.stats.sac['stlo'],
                                                   origin.latitude,
                                                   origin.longitude)

            distance *= km2deg

            if dist_min < distance < dist_max:

                # Calculate travel times for the data
                # Here we use the origin depth.
                tt = model.get_travel_times(distance_in_degree=distance,
                                            source_depth_in_km=origin.depth *
                                            1e-3,
                                            phase_list=phase_list)
                travel_time_data = origin.time + tt[0].time

                # Calculate travel times for the synthetics
                # Here we use the inversion depth.
                tt = model.get_travel_times(distance_in_degree=distance,
                                            source_depth_in_km=depth_in_m *
                                            1e-3,
                                            phase_list=phase_list)
                travel_time_synth = origin.time + tt[0].time

                # Trim data around P arrival time
                tr_work.trim(starttime=travel_time_data - pre_offset,
                             endtime=travel_time_data + post_offset)

                lendiff = tr_work.stats.npts - npts_target

                if lendiff == -1:
                    tr_work.data = np.r_[tr_work.data, 0]
                elif lendiff == 1:
                    tr_work.data = tr_work.data[0:-1]
                elif abs(lendiff) > 1:
                    raise IndexError('Difference in length too big')
                st_data.append(tr_work)

                # Get synthetics
                # See what exists in the stream that we loaded earlier
                gf6_synth = st_synth_grf6.select(station=tr.stats.station,
                                                 network=tr.stats.network,
                                                 location=tr.stats.location)

                if gf6_synth:
                    # len(gf6_synth == 6):
                    # Data already exists
                    st_synth += gf6_synth[0:6]
                else:
                    # Data does not exist yet
                    gf_synth = Stream()
                    gf_synth += db.get_greens_function(distance,
                                                       depth_in_m,
                                                       dt=tr_work.stats.delta)
                    for tr_synth in gf_synth:
                        tr_synth.stats['starttime'] = \
                            tr_synth.stats.starttime + float(origin.time)

                        tr_synth.trim(starttime=travel_time_synth - pre_offset,
                                      endtime=travel_time_synth + post_offset)

                        lendiff = tr_synth.stats.npts - npts_target
                        if lendiff == -1:
                            tr_synth.data = np.r_[tr_synth.data, 0]
                        elif lendiff == 1:
                            tr_synth.data = tr_synth.data[0:-1]
                        elif abs(lendiff) > 1:
                            raise IndexError('Difference in length too big')

                    # Convert Green's functions from seiscomp format to one per
                    # MT component, which is used later in the inversion.
                    # Convert to GRF6 format
                    st_synth += gf_synth.seiscomp_to_moment_tensor(
                        azimuth=azi,
                        scalmom=1,
                        stats=tr_work.stats)

        for tr in st_synth:
            tr.write(os.path.join(grf6_dir, 'synth_%06dkm_%s.SAC' %
                                  (depth_in_m, tr.id)),
                     format='SAC')

        for tr in st_data:
            tr.write(os.path.join(data_dir, 'data_%s.SAC' % tr.id),
                     format='SAC')

        print('%d stations requested, %d were in range for phase %s' %
              (len(self), len(st_data), phase_list))

        return st_data, st_synth

    def calc_timeshift(self, st_b, allow_negative_CC=False):
        """
        dt_all, CC = calc_timeshift(self, st_b, allow_negative_CC)

        Calculate timeshift between two waveforms using the maximum of the
        cross-correlation function.

        Parameters
        ----------
        self : obspy.Stream
            Stream that contains the reference traces

        st_b : obspy.Stream
            Stream that contains the traces to compare

        allow_negative_CC : boolean
            Pick the maximum of the absolute values of CC(t). Useful,
            if polarity may be wrong.

        Returns
        -------
        dt_all : dict
            Dictionary with entries station.location and the estimated
            time shift in seconds.

        CC : dict
            Dictionary with the correlation coefficients for each station.

        """
        dt_all = dict()
        CC_all = dict()
        for tr_a in self:
            try:
                tr_b = st_b.select(station=tr_a.stats.station,
                                   location=tr_a.stats.location)[0]
                corr = signal.correlate(tr_a.data, tr_b.data)

                if allow_negative_CC:
                    idx_CCmax = np.argmax(abs(corr))
                    CC = abs(corr[idx_CCmax])
                else:
                    idx_CCmax = np.argmax(corr)
                    CC = corr[idx_CCmax]

                dt = (idx_CCmax - tr_a.stats.npts + 1) * tr_a.stats.delta

                CC /= np.sqrt(np.sum(tr_a.data**2) * np.sum(tr_b.data**2))
                # print('%s.%s: %4.1f sec, CC: %f' %
                #       (tr_a.stats.station, tr_a.stats.location, dt, CC))
                code = '%s.%s' % (tr_a.stats.station, tr_a.stats.location)
                dt_all[code] = dt
                CC_all[code] = CC
            except IndexError:
                print('Did not find %s' % (tr_a.stats.station))
        return dt_all, CC_all

    def calc_amplitude_misfit(self, st_b):
        """
        dA_all = calc_amplitude_misfit(self, st_b)

        Calculate amplitude misfit between two waveforms as defined in Dahlen &
        Baig (2002).

        Parameters
        ----------
        self : obspy.Stream
            Stream that contains the reference traces (usually data)

        st_b : obspy.Stream
            Stream that contains the traces to compare (usually synthetic)


        Returns
        -------
        dA_all : dict
            Dictionary with entries station.location and the estimated
            amplitude misfit.

        """
        dA_all = dict()

        for tr_a in self:
            try:
                tr_b = st_b.select(station=tr_a.stats.station,
                                   location=tr_a.stats.location)[0]

                if abs(tr_a.stats.npts - tr_b.stats.npts) > 1:
                    raise ValueError('Lengths of traces differ by more than \
                                     one sample, %d vs %d samples' %
                                     (tr_a.stats.npts, tr_b.stats.npts))
                elif abs(tr_a.stats.npts - tr_b.stats.npts) == 1:
                    len_common = min(tr_a.stats.npts, tr_b.stats.npts)
                else:
                    len_common = tr_a.stats.npts

                dA = abs(np.sum(tr_a.data[0:len_common] *
                                tr_b.data[0:len_common])) / \
                    np.sum(tr_b.data ** 2)

                code = '%s.%s' % (tr_a.stats.station, tr_a.stats.location)
                dA_all[code] = dA
            except IndexError:
                print('Did not find %s' % (tr_a.stats.station))
        return dA_all

    def calc_L2_misfit(self, st_b):
        L2 = 0
        for tr_a in self:
            try:
                tr_b = st_b.select(station=tr_a.stats.station,
                                   location=tr_a.stats.location)[0]

                if abs(tr_a.stats.npts - tr_b.stats.npts) > 1:
                    raise ValueError('Lengths of traces differ by more than \
                                    one sample')
                elif abs(tr_a.stats.npts - tr_b.stats.npts) == 1:
                    len_common = min(tr_a.stats.npts, tr_b.stats.npts)
                else:
                    len_common = tr_a.stats.npts

                RMS = np.sum((tr_a.data[0:len_common] -
                              tr_b.data[0:len_common]) ** 2)

                L2 += RMS
            except IndexError:
                print('Did not find %s' % (tr_a.stats.station))
        return np.sqrt(L2) / len(self)

    def shift_waveform(self, dt):
        # Create new stream with time-shifted synthetic seismograms
        for tr in self:
            code = '%s.%s' % (tr.stats.station, tr.stats.location)
            shift_waveform(tr, dt[code])

        return

    def taper_before_arrival(self, st_synth, threshold=None):
        """
        taper_before_arrival(self, st_synth)

        Taper data in stream based on the arrival time of seismic
        energy in the seismograms in the second stream.


        Parameters
        ----------
        self: stfinv.Stream
            Stream with measured data. Tapering is done in place.

        st_synth : obspy.Stream
            Stream with synthetic data. Serves as the reference to determine
            tapering time window.


        Returns
        -------
        None

        """

        len_win = 0.0
        arr_times = dict()

        for tr in self:
            tr_synth = st_synth.select(station=tr.stats.station,
                                       network=tr.stats.network,
                                       location=tr.stats.location)[0]
            if not threshold:
                threshold = max(abs(tr_synth.data)) * 1e-2
            arr_time = pick(tr_synth, threshold=threshold)
            taper_signal(tr, t_begin=arr_time, t_end=arr_time + 30.0)

            len_win = max(len_win, 30.0)
            arr_times['%s.%s' % (tr.stats.station, tr.stats.location)] = \
                arr_time

        return len_win, arr_times

    def calc_synthetic_from_grf6(self, st_data, stf, tensor):
        st_synth = st_data.copy()

        for tr in st_synth:
            stat = tr.stats.station
            loc = tr.stats.location
            tr.data = (self.select(station=stat,
                                   location=loc,
                                   channel='MTT')[0].data * tensor.m_tt +
                       self.select(station=stat,
                                   location=loc,
                                   channel='MPP')[0].data * tensor.m_pp +
                       self.select(station=stat,
                                   location=loc,
                                   channel='MRR')[0].data * tensor.m_rr +
                       self.select(station=stat,
                                   location=loc,
                                   channel='MTP')[0].data * tensor.m_tp +
                       self.select(station=stat,
                                   location=loc,
                                   channel='MRT')[0].data * tensor.m_rt +
                       self.select(station=stat,
                                   location=loc,
                                   channel='MRP')[0].data * tensor.m_rp)
            # Convolve with STF
            tr.data = np.convolve(tr.data, stf, mode='same')[0:tr.stats.npts]

        return st_synth


def read(path):
    stream = Stream()
    stream += obspy.read(path)
    return stream


def load_cut_files(data_directory, grf6_directory, depth_in_m):
    # This can correctly retrieve the channel names of the grf6 synthetics
    # Load grf6 synthetics
    files_synth = glob.glob(os.path.join(grf6_directory,
                                         'synth_%06dkm*' % depth_in_m))
    files_synth.sort()

    st_synth_grf = Stream()
    for file_synth in files_synth:
        tr = obspy.read(file_synth)[0]
        tr.stats.channel = file_synth.split('.')[-2]
        st_synth_grf.append(tr)

    # Load data
    files_data = glob.glob(os.path.join(data_directory, 'data*'))
    files_data.sort()

    st_data = Stream()
    for file_data in files_data:
        st_data.append(obspy.read(file_data)[0])

    return st_data, st_synth_grf


def taper_signal(trace, t_begin, t_end):
    """
    taper_signal(self, t_begin, t_end)

    Taper data array in self with an asymmetric Hanning window. The range
    between t_begin and t_end is left unchanged. The two seconds before t_begin
    are tapered with the rising half of a Hanning window. The (t_end-t_begin)
    seconds after t_end are tapered with the decaying half of a Hanning window.


    Parameters
    ----------
    trace : obspy.Trace
        ObsPy Trace object with signal.

    t_begin : Begin of signal window

    t_end : End of signal window


    Returns
    -------
    None
        Note that the data array in the self is modified in place.

    """
    window = np.zeros_like(trace.data)

    times = trace.times() - float(trace.times()[0])

    i_begin = abs(times - (t_begin - 2.0)).argmin()
    i_end = abs(times - t_end).argmin()

    winlen_begin = int(2.0 / trace.stats.delta)
    winlen_end = int((t_end - t_begin) / trace.stats.delta)

    # Signal part from zero to 2s before t_begin is muted
    window[0:i_begin] = 0.0

    # Signal part two seconds before t_begin is tapered with Hanning function
    window[i_begin:i_begin + winlen_begin] = \
        np.hanning(winlen_begin * 2)[0:winlen_begin]

    # Signal part between t_begin and t_end is left unchanged.
    window[i_begin + winlen_begin:i_end] = 1

    # Signal part after t_end is tapered with Hanning function with width
    # (t_end-t_begin).
    if i_end + winlen_end < len(window):
        window[i_end:i_end + winlen_end] = \
            np.hanning(winlen_end * 2)[winlen_end:]
    else:
        remainder = len(window) - i_end
        window[i_end:] = \
            np.hanning(winlen_end * 2)[winlen_end:winlen_end + remainder]

    trace.data *= window


def pick(trace, threshold):
    """
    first_break = pick(signal, threshold)

    Return first time of signal, which crosses threshold from below.
    Note that the first crossing is returned, so if signal[0] is above
    threshold, this does not count.

    Parameters
    ----------
    signal : np.array
        Array with signal.

    threshold : float
        Threshold


    Returns
    -------
    first_breaki : float
        Time of first crossing of threshold

    """
    thresholded_data = abs(trace.data) > threshold
    threshold_edges = np.convolve([1, -1], thresholded_data, mode='same')
    threshold_edges[0] = 0

    crossings = np.where(threshold_edges == 1)
    if crossings[0].any():
        first_break = trace.times()[crossings[0][0]]
    else:
        first_break = trace.times()[0]

    first_break -= float(trace.times()[0])

    return first_break


def shift_waveform(tr, dtshift):
    """
    tr_shift = shift_waveform(tr, dtshift):

    Shift data in trace tr by dtshift seconds backwards.

    Parameters
    ----------
    tr : obspy.Trace
        Trace that contains the data to shift

    dtshift : float
        Time shift in seconds


    Returns
    -------
    tr_shift : obspy.Trace
        Copy of tr, but with data shifted dtshift seconds backwards.

    """
    data_pad = np.r_[tr.data, np.zeros_like(tr.data)]

    freq = fft.fftfreq(len(data_pad), tr.stats.delta)
    shiftvec = np.exp(- 2 * np.pi * complex(0., 1.) * freq * dtshift)
    data_fd = shiftvec * fft.fft(data_pad *
                                 signal.tukey(len(data_pad),
                                              alpha=0.2))

    tr.data = np.real(fft.ifft(data_fd))[0:tr.stats.npts]
