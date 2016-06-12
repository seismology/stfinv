import numpy as np
import obspy
import glob
import os
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from scipy import signal
import scipy.fftpack as fft
import matplotlib.pyplot as plt


__all__ = ["seiscomp_to_moment_tensor",
           "get_synthetics",
           "shift_waveform",
           "calc_timeshift",
           "calc_amplitude_misfit",
           "calc_L2_misfit",
           "create_matrix_MT_inversion",
           "filter_bad_waveforms",
           "invert_MT",
           "load_cut_files",
           "calc_synthetic_from_grf6",
           "plot_waveforms",
           "correct_and_remove_bad_waveforms",
           "get_station_coordinates"]


def seiscomp_to_moment_tensor(st_in, azimuth, stats=None, scalmom=1.0):
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
    m_tt = st_in.select(channel='ZSS')[0].data * np.cos(2 * azimuth) / 2. + \
        st_in.select(channel='ZEP')[0].data / 3. -                     \
        st_in.select(channel='ZDD')[0].data / 6.
    m_tt *= scalmom

    m_pp = - st_in.select(channel='ZSS')[0].data * np.cos(2 * azimuth) / 2. + \
        st_in.select(channel='ZEP')[0].data / 3. -                            \
        st_in.select(channel='ZDD')[0].data / 6.
    m_pp *= scalmom

    m_rr = st_in.select(channel='ZEP')[0].data / 3. + \
        st_in.select(channel='ZDD')[0].data / 3.
    m_rr *= scalmom

    m_tp = st_in.select(channel='ZSS')[0].data * np.sin(2 * azimuth)
    m_tp *= scalmom

    m_rt = - st_in.select(channel='ZDS')[0].data * np.cos(azimuth)
    m_rt *= scalmom

    m_rp = st_in.select(channel='ZDS')[0].data * np.sin(azimuth)
    m_rp *= scalmom

    data = [m_tt, m_pp, m_rr, -m_tp, m_rt, m_rp]
    channels = ['MTT', 'MPP', 'MRR', 'MTP', 'MRT', 'MRP']

    st_grf6 = obspy.Stream()
    for icomp in range(0, 6):
        if stats:
            tr_new = obspy.Trace(data=data[icomp],
                                 header=stats)
        else:
            tr_new = obspy.Trace(data=data[icomp])
        tr_new.stats['channel'] = channels[icomp]
        st_grf6.append(tr_new)

    return st_grf6


def get_synthetics(stream, origin, db, pre_offset=5.6, post_offset=20.0,
                   dist_min=30.0, dist_max=85.0, phase_list='P',
                   outdir_data='data', outdir_grf6='grf6'):

    km2deg = 360.0 / (2 * np.pi * 6378137.0)

    model = TauPyModel(model="iasp91")

    st_data = obspy.Stream()
    st_synth = obspy.Stream()

    for tr in stream:
        tr_work = tr.copy()

        distance, azi, bazi = gps2dist_azimuth(tr.stats.sac['stla'],
                                               tr.stats.sac['stlo'],
                                               origin.latitude,
                                               origin.longitude)
        distance *= km2deg

        if dist_min < distance < dist_max:
            tt = model.get_travel_times(distance_in_degree=distance,
                                        source_depth_in_km=origin.depth * 1e-3,
                                        phase_list=phase_list)
            travel_time = origin.time + tt[0].time

            # print('%6s, %8.3f degree, %8.3f sec\n' % (tr.stats.station,
            #                                           distance, travel_time))

            # Trim data around P arrival time
            tr_work.trim(starttime=travel_time - pre_offset,
                         endtime=travel_time + post_offset)

            st_data.append(tr_work)

            # Get synthetics
            gf_synth = db.get_greens_function(distance,
                                              source_depth_in_m=origin.depth,
                                              dt=tr_work.stats.delta)
            for tr_synth in gf_synth:
                tr_synth.stats['starttime'] = tr_synth.stats.starttime + \
                    float(origin.time)

                tr_synth.trim(starttime=travel_time - pre_offset,
                              endtime=travel_time + post_offset)

            # Convert Green's functions from seiscomp format to one per MT
            # component, which is used later in the inversion.
            # Convert to GRF6 format
            st_synth += seiscomp_to_moment_tensor(gf_synth,
                                                  azimuth=azi,
                                                  scalmom=1,
                                                  stats=tr_work.stats)

        else:
            print('%6s, %8.3f degree, out of range\n' %
                  (tr.stats.station, distance))

    for tr in st_synth:
        tr.write(os.path.join(outdir_data, 'synth_%s.SAC' % tr.id),
                 format='SAC')

    for tr in st_data:
        tr.write(os.path.join(outdir_grf6, 'data_%s.SAC' % tr.id),
                 format='SAC')

    print('%d stations requested, %d were in range for phase %s' %
          (len(stream), len(st_data), phase_list))

    return st_data, st_synth


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
    tr_shift = tr.copy()

    freq = fft.fftfreq(tr.stats.npts, tr.stats.delta)
    shiftvec = np.exp(- 2 * np.pi * complex(0., 1.) * freq * dtshift)
    data_fd = shiftvec * fft.fft(tr_shift.data *
                                 signal.tukey(tr_shift.stats.npts,
                                              alpha=0.2))

    tr_shift.data = np.real(fft.ifft(data_fd))
    return tr_shift


def calc_timeshift(st_a, st_b):
    """
    dt_all = calc_timeshift(st_a, st_b)

    Calculate timeshift between two waveforms using the maximum of the
    cross-correlation function.

    Parameters
    ----------
    st_a : obspy.Stream
        Stream that contains the reference traces

    st_b : obspy.Stream
        Stream that contains the traces to compare


    Returns
    -------
    dt_all : dict
        Dictionary with entries station.location and the estimated time shift
        in seconds.

    CC : dict
        Dictionary with the correlation coefficients for each station.

    """
    dt_all = dict()
    CC_all = dict()
    for tr_a in st_a:
        try:
            tr_b = st_b.select(station=tr_a.stats.station,
                               location=tr_a.stats.location)[0]
            corr = signal.correlate(tr_a.data, tr_b.data)
            dt = (np.argmax(corr) - tr_a.stats.npts + 1) * tr_a.stats.delta
            CC = corr[np.argmax(corr)] / np.sqrt(np.sum(tr_a.data**2) *
                                                 np.sum(tr_b.data**2))
            print('%s.%s: %4.1f sec, CC: %f' %
                  (tr_a.stats.station, tr_a.stats.location, dt, CC))
            dt_all['%s.%s' % (tr_a.stats.station, tr_a.stats.location)] = dt
            CC_all['%s.%s' % (tr_a.stats.station, tr_a.stats.location)] = CC
        except IndexError:
            print('Did not find %s' % (tr_a.stats.station))
    return dt_all, CC_all


def calc_amplitude_misfit(st_a, st_b):
    """
    dA_all = calc_amplitude_misfit(st_a, st_b)

    Calculate amplitude misfit between two waveforms as defined in Dahlen &
    Baig (2002).

    Parameters
    ----------
    st_a : obspy.Stream
        Stream that contains the reference traces (usually data)

    st_b : obspy.Stream
        Stream that contains the traces to compare (usually synthetic)


    Returns
    -------
    dA_all : dict
        Dictionary with entries station.location and the estimated amplitude
        misfit.

    """
    dA_all = dict()
    for tr_a in st_a:
        try:
            tr_b = st_b.select(station=tr_a.stats.station,
                               location=tr_a.stats.location)[0]
            dA = np.sum(tr_a.data * tr_b.data) / np.sum(tr_b.data ** 2)

            print('%s.%s: %4.2f ' %
                  (tr_a.stats.station, tr_a.stats.location, dA))
            dA_all['%s.%s' % (tr_a.stats.station, tr_a.stats.location)] = dA
        except IndexError:
            print('Did not find %s' % (tr_a.stats.station))
    return dA_all


def calc_L2_misfit(st_a, st_b):
    L2 = 0
    for tr_a in st_a:
        try:
            tr_b = st_b.select(station=tr_a.stats.station,
                               location=tr_a.stats.location)[0]
            RMS = np.sum((tr_a.data - tr_b.data) ** 2)

            # print('%s.%s: %e ' %
            #       (tr_a.stats.station, tr_a.stats.location, RMS))
            L2 += RMS
        except IndexError:
            print('Did not find %s' % (tr_a.stats.station))
    return np.sqrt(L2)


def create_matrix_MT_inversion(st_data, st_grf6):
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


def filter_bad_waveforms(stream, CC, CClim):
    # Create new stream with time-shifted synthetic seismograms
    st_filtered = obspy.Stream()
    for tr in stream:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        if CC[code] > CClim:
            st_filtered.append(tr)
    return st_filtered


def invert_MT(st_data, st_grf6, tens_orig, outdir='focmec'):
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

    tens_orig : obspy.core.event.Tensor
        Previous moment tensor


    Returns
    -------
    tens_new : obspy.core.event.Tensor
        Updated moment tensor

    """

    d, G = create_matrix_MT_inversion(st_data, st_grf6)

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
    m_orig = [tens_orig.m_rr,
              tens_orig.m_tt,
              tens_orig.m_pp,
              tens_orig.m_rt,
              tens_orig.m_rp,
              tens_orig.m_tp]

    m_est = [tens_new.m_rr,
             tens_new.m_tt,
             tens_new.m_pp,
             tens_new.m_rt,
             tens_new.m_rp,
             tens_new.m_tp]

    fig = plt.figure(figsize=(5, 10))
    obspy.imaging.beachball.beachball(fm=m_orig, xy=(-100, 0),
                                      fig=fig)

    obspy.imaging.beachball.beachball(fm=m_est, xy=(100, 0),
                                      fig=fig)
    fig.savefig(os.path.join(outdir, 'bb.png'))
    return tens_new


def load_cut_files(directory):
    # This can correctly retrieve the channel names of the grf6 synthetics
    # Load grf6 synthetics
    files_synth = glob.glob(os.path.join(directory, 'synth*'))
    files_synth.sort()

    st_synth_grf = obspy.Stream()
    for file_synth in files_synth:
        tr = obspy.read(file_synth)[0]
        tr.stats.channel = file_synth.split('.')[-2]
        st_synth_grf.append(tr)

    # Load data
    files_data = glob.glob(os.path.join(directory, 'data*'))
    files_data.sort()

    st_data = obspy.Stream()
    for file_data in files_data:
        st_data.append(obspy.read(file_data)[0])

    return st_data, st_synth_grf


def calc_synthetic_from_grf6(st_synth_grf6, st_data, tensor):
    st_synth = st_data.copy()

    for tr in st_synth:
        stat = tr.stats.station
        loc = tr.stats.location
        tr.data = (st_synth_grf6.select(station=stat,
                                        location=loc,
                                        channel='MTT')[0].data * tensor.m_tt +
                   st_synth_grf6.select(station=stat,
                                        location=loc,
                                        channel='MPP')[0].data * tensor.m_pp +
                   st_synth_grf6.select(station=stat,
                                        location=loc,
                                        channel='MRR')[0].data * tensor.m_rr +
                   st_synth_grf6.select(station=stat,
                                        location=loc,
                                        channel='MTP')[0].data * tensor.m_tp +
                   st_synth_grf6.select(station=stat,
                                        location=loc,
                                        channel='MRT')[0].data * tensor.m_rt +
                   st_synth_grf6.select(station=stat,
                                        location=loc,
                                        channel='MRP')[0].data * tensor.m_rp)
    return st_synth


def plot_waveforms(st_data, st_synth, outdir='./pic_corr/'):

    for tr in st_data:
        plt.plot(st_synth.select(station=tr.stats.station)[0].times(),
                 st_synth.select(station=tr.stats.station)[0].data,
                 label='Synthetic, corrected')
        plt.plot(tr.times(),
                 tr.data, label='data')
        plt.legend()
        plt.title('%s' % tr.stats.station)
        outfile = os.path.join(outdir, '%s.png' % tr.stats.station)
        plt.savefig(outfile)
        plt.close()


def correct_and_remove_bad_waveforms(st_data, st_synth, st_synth_grf6,
                                     CCmin=0.75):
    # Create working copies of streams
    st_data_work = st_data.copy()
    st_synth_work = st_synth.copy()
    st_synth_grf6_work = st_synth_grf6.copy()

    # Calculate time shift from combined synthetic waveforms
    dt, CC = calc_timeshift(st_data_work, st_synth_work)

    # Remove bad waveforms
    st_synth_work = filter_bad_waveforms(st_synth_work, CC, CCmin)
    st_data_work = filter_bad_waveforms(st_data_work, CC, CCmin)
    st_synth_grf6_work = filter_bad_waveforms(st_synth_grf6_work, CC, CCmin)

    # Create new stream with time-shifted synthetic seismograms
    st_synth_shift = obspy.Stream()
    for tr in st_synth_work:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        tr_new = shift_waveform(tr, dt[code])
        st_synth_shift.append(tr_new)

    # Create new stream with time-shifted Green's functions
    st_synth_grf6_shift = obspy.Stream()
    for tr in st_synth_grf6_work:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        tr_new = shift_waveform(tr, dt[code])
        st_synth_grf6_shift.append(tr_new)

    # Calculate amplitude misfit
    dA = calc_amplitude_misfit(st_data_work, st_synth_shift)

    # Create new stream with amplitude-corrected synthetic seismograms
    for tr in st_synth_shift:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        tr.data *= dA[code]

    # Create new stream with amplitude-corrected Green's functions
    for tr in st_synth_grf6_shift:
        tr.data *= dA[code]

    return st_data_work, st_synth_shift, st_synth_grf6_shift, CC, dt, dA


def get_station_coordinates(stream, client_base_url='IRIS'):
    # Attaches station coordinates to the traces of the stream
    # They are written into the stats.sac dictionary
    from obspy.clients.fdsn import Client
    client = Client(client_base_url)
    bulk = []
    for tr in stream:
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

    for tr in stream:
        stats = tr.stats
        stat = inv.select(network=stats.network,
                          station=stats.station,
                          location=stats.location,
                          channel=stats.channel)
        stats.sac = dict(stla=stat[0][0].latitude,
                         stlo=stat[0][0].longitude)
