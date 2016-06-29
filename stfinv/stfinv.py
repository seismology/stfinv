import numpy as np
import obspy
import glob
import os
import instaseis
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from obspy.imaging.beachball import beach
from scipy import signal, linalg
import scipy.fftpack as fft
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
import argparse


__all__ = ["inversion",
           "seiscomp_to_moment_tensor",
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
           "correct_waveforms",
           "get_station_coordinates",
           "create_Toeplitz",
           "create_Toeplitz_mult",
           "create_matrix_STF_inversion",
           "invert_STF",
           "pick",
           "taper_signal",
           "taper_before_arrival"]


def inversion(data_path, event_file, db_path='syngine://ak135f_2s',
              depth_in_m=-1, dist_min=30.0, dist_max=100.0, CClim=0.6,
              phase_list=('P', 'Pdiff'),
              pre_offset=15,
              post_offset=36.1,
              work_dir='testinversion'):
    from obspy.signal.interpolation import lanczos_interpolation

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # Instaseis does not like sources at 0.0 km
    if depth_in_m == 0.0:
        depth_in_m += 0.01

    print('Inverting for depth %5.2fkm' % (depth_in_m * 1e-3))

    # Read all data
    st = obspy.read(os.path.join(data_path, '*'))

    # Fill stream with station coordinates (in SAC header)
    get_station_coordinates(st)

    # Read event file
    cat = obspy.read_events(event_file)
    event = cat[0]

    db = instaseis.open_db(db_path)

    # Calculate synthetics in GRF6 format with instaseis and cut
    # time windows around the phase arrivals (out of data and GRF6 synthetics.
    # The cuts are saved on disk for the next time.
    st_data, st_synth_grf6 = get_synthetics(st,
                                            event.origins[0],
                                            db,
                                            depth_in_m=depth_in_m,
                                            out_dir=work_dir,
                                            pre_offset=pre_offset,
                                            post_offset=post_offset,
                                            dist_min=dist_min,
                                            dist_max=dist_max,
                                            phase_list=phase_list)

    # Convolve st_data with Instaseis stf to remove its effect.
    sliprate = lanczos_interpolation(db.info.slip, old_start=0,
                                     old_dt=db.info.dt, new_dt=0.1,
                                     new_start=0,
                                     new_npts=st_data[0].stats.npts,
                                     a=8)
    for tr in st_data:
        tr.data = np.convolve(tr.data, sliprate)[0:tr.stats.npts]

    # Start values to ensure one iteration
    it = 0
    misfit_reduction = 1e8
    misfit_new = 2

    # Initialize with MT from event file
    try:
        tensor = cat[0].focal_mechanisms[0].moment_tensor.tensor
    except IndexError:
        print('No moment tensor present, using explosion. Hilarity may ensue')
        tensor = obspy.core.event.Tensor(m_rr=1e20, m_tt=1e20, m_pp=1e20,
                                         m_rp=0.0, m_rt=0.0, m_tp=0.0)

    # Init with spike STF
    stf = np.zeros(128)
    stf[1] = 1.

    # Define butterworth filter at database corner frequency
    b, a = signal.butter(6, Wn=((1. / (db.info.dt * 2.)) /
                                (1. / 0.2)))

    while misfit_reduction > -0.1:
        # Get synthetics for current source solution
        st_synth = calc_synthetic_from_grf6(st_synth_grf6,
                                            st_data,
                                            stf=stf,
                                            tensor=tensor)
        st_data_work = st_data.copy()
        st_data_work, st_synth_corr, \
            st_synth_grf6_corr, CC, dT, dA = correct_waveforms(st_data_work,
                                                               st_synth,
                                                               st_synth_grf6)

        len_win, arr_times = taper_before_arrival(st_data_work,
                                                  st_synth_corr)
        len_win, arr_times = taper_before_arrival(st_synth_grf6_corr,
                                                  st_synth_corr)

        nstat_used = len(filter_bad_waveforms(st_data_work,
                                              CC, CClim))

        # Calculate misfit reduction
        misfit_old = misfit_new
        misfit_new = calc_D_misfit(CC)
        misfit_reduction = (misfit_old - misfit_new) / misfit_old

        print('  it: %02d, misfit: %5.3f (%8.1f pct red. %d stations)' %
              (it, misfit_new, misfit_reduction * 1e2, nstat_used))

        plot_waveforms(st_data_work, st_synth_corr,
                       arr_times, CC, CClim, dA, dT,
                       outdir=os.path.join(work_dir,
                                           'waveforms_%06dkm' % depth_in_m),
                       misfit=misfit_new,
                       iteration=it, stf=stf, tensor=tensor)

        # Stop the inversion, if no station can be used
        if (nstat_used == 0):
            break

        # Omit the STF inversion in the first round.
        if it > 0:
            st_data_stf = filter_bad_waveforms(st_data_work,
                                               CC, CClim).copy()

            # Do the STF inversion on downsampled data to avoid high frequency
            # noise that has to be removed later
            st_data_stf.decimate(factor=5)
            st_synth_stf = filter_bad_waveforms(st_synth_corr,
                                                CC, CClim).copy()
            st_synth_stf.decimate(factor=5)
            stf = invert_STF(st_data_stf, st_synth_stf)

            stf = lanczos_interpolation(stf,
                                        old_start=0, old_dt=0.5,
                                        new_start=0, new_dt=0.1,
                                        new_npts=128,
                                        a=8)

        tensor = invert_MT(filter_bad_waveforms(st_data_work,
                                                CC, CClim),
                           filter_bad_waveforms(st_synth_grf6_corr,
                                                CC, CClim),
                           stf=stf,
                           outdir=os.path.join(work_dir, 'focmec'))
        it += 1


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


def get_synthetics(stream, origin, db, out_dir='inversion',
                   depth_in_m=-1,
                   pre_offset=5.6, post_offset=20.0,
                   dist_min=30.0, dist_max=85.0, phase_list='P'):

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

            # Calculate travel times for the data
            # Here we use the origin depth.
            tt = model.get_travel_times(distance_in_degree=distance,
                                        source_depth_in_km=origin.depth * 1e-3,
                                        phase_list=phase_list)
            travel_time_data = origin.time + tt[0].time

            # Calculate travel times for the synthetics
            # Here we use the inversion depth.
            tt = model.get_travel_times(distance_in_degree=distance,
                                        source_depth_in_km=depth_in_m * 1e-3,
                                        phase_list=phase_list)
            travel_time_synth = origin.time + tt[0].time

            # print('%6s, %8.3f degree, %8.3f sec\n' % (tr.stats.station,
            #                                           distance, travel_time))

            # Trim data around P arrival time
            tr_work.trim(starttime=travel_time_data - pre_offset,
                         endtime=travel_time_data + post_offset)

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
                gf_synth = db.get_greens_function(distance,
                                                  source_depth_in_m=depth_in_m,
                                                  dt=tr_work.stats.delta)
                for tr_synth in gf_synth:
                    tr_synth.stats['starttime'] = tr_synth.stats.starttime + \
                        float(origin.time)

                    tr_synth.trim(starttime=travel_time_synth - pre_offset,
                                  endtime=travel_time_synth + post_offset)

                # Convert Green's functions from seiscomp format to one per MT
                # component, which is used later in the inversion.
                # Convert to GRF6 format
                st_synth += seiscomp_to_moment_tensor(gf_synth,
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

    data_pad = np.r_[tr_shift.data, np.zeros_like(tr_shift.data)]

    freq = fft.fftfreq(len(data_pad), tr.stats.delta)
    shiftvec = np.exp(- 2 * np.pi * complex(0., 1.) * freq * dtshift)
    data_fd = shiftvec * fft.fft(data_pad *
                                 signal.tukey(len(data_pad),
                                              alpha=0.2))

    tr_shift.data = np.real(fft.ifft(data_fd))[0:tr_shift.stats.npts]
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
            # print('%s.%s: %4.1f sec, CC: %f' %
            #       (tr_a.stats.station, tr_a.stats.location, dt, CC))
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

            if abs(tr_a.stats.npts - tr_b.stats.npts) > 1:
                raise ValueError('Lengths of traces differ by more than \
                                  one sample')
            elif abs(tr_a.stats.npts - tr_b.stats.npts) == 1:
                len_common = min(tr_a.stats.npts, tr_b.stats.npts)
            else:
                len_common = tr_a.stats.npts

            dA = abs(np.sum(tr_a.data[0:len_common] *
                            tr_b.data[0:len_common])) / \
                np.sum(tr_b.data ** 2)

            dA_all['%s.%s' % (tr_a.stats.station, tr_a.stats.location)] = dA
        except IndexError:
            print('Did not find %s' % (tr_a.stats.station))
    return dA_all


def calc_D_misfit(CCs):
    CC = []
    for key, value in CCs.items():
        CC.append(1. - value)

    return np.mean(CC)


def calc_L2_misfit(st_a, st_b):
    L2 = 0
    for tr_a in st_a:
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
    return np.sqrt(L2) / len(st_a)


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
        # print('%s, %f' % (code, CC[code]))
        if CC[code] > CClim:
            # print('in')
            st_filtered.append(tr)
        # else:
        #    print('out')
    return st_filtered


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
        tr.data = np.convolve(tr.data, stf, mode='same')

    d, G = create_matrix_MT_inversion(st_data, st_grf6_work)

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


def load_cut_files(data_directory, grf6_directory, depth_in_m):
    # This can correctly retrieve the channel names of the grf6 synthetics
    # Load grf6 synthetics
    files_synth = glob.glob(os.path.join(grf6_directory,
                                         'synth_%06dkm*' % depth_in_m))
    files_synth.sort()

    st_synth_grf = obspy.Stream()
    for file_synth in files_synth:
        tr = obspy.read(file_synth)[0]
        tr.stats.channel = file_synth.split('.')[-2]
        st_synth_grf.append(tr)

    # Load data
    files_data = glob.glob(os.path.join(data_directory, 'data*'))
    files_data.sort()

    st_data = obspy.Stream()
    for file_data in files_data:
        st_data.append(obspy.read(file_data)[0])

    return st_data, st_synth_grf


def calc_synthetic_from_grf6(st_synth_grf6, st_data, stf, tensor):
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
        # Convolve with STF
        tr.data = np.convolve(tr.data, stf, mode='same')

    return st_synth


def plot_waveforms(st_data, st_synth, arr_times, CC, CClim, dA, dT, stf,
                   tensor, iteration=-1, misfit=0.0, outdir='./waveforms/'):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    nplots = len(st_data)
    nrows = int(np.sqrt(nplots)) + 1
    ncols = nplots / nrows + 1
    iplot = 0
    for tr in st_data:

        irow = np.mod(iplot, nrows)
        icol = np.int(iplot / nrows)

        normfac = max(np.abs(tr.data))

        yoffset = irow * 1.5
        xoffset = icol

        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        if CC[code] > CClim:
            ls = '-'
        else:
            ls = 'dotted'

        yvals = st_synth.select(station=tr.stats.station)[0].data / normfac
        xvals = np.linspace(0, 0.8, num=len(yvals))
        l_s, = ax.plot(xvals + xoffset,
                       yvals + yoffset,
                       color='r',
                       linestyle=ls,
                       linewidth=2)

        yvals = tr.data / normfac
        xvals = np.linspace(0, 0.8, num=len(yvals))
        l_d, = ax.plot(xvals + xoffset,
                       yvals + yoffset,
                       color='k',
                       linestyle=ls,
                       linewidth=1.5)
        ax.text(xoffset, yoffset + 0.2,
                '%s \nCC: %4.2f\ndA: %4.1f\ndT: %5.1f' % (tr.stats.station,
                                                          CC[code],
                                                          dA[code],
                                                          dT[code]),
                size=8.0, color='darkgreen')

        xvals = ((arr_times[code] / tr.times()[-1]) * 0.8 + xoffset) * \
            np.ones(2)
        ax.plot(xvals, (yoffset + 0.5, yoffset - 0.5), 'b')

        iplot += 1

    ax.legend((l_s, l_d), ('Synthetic', 'data'))
    ax.set_xlim(0, ncols * 1.2)

    if (iteration >= 0):
        ax.set_title('Waveform fits, iteration %d, misfit: %9.3e' %
                     (iteration, misfit))

    # Plot STF
    left, bottom, width, height = [0.7, 0.2, 0.18, 0.18]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(stf)
    ax2.set_ylim((-0.2, 1.1))
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot beach ball
    mt = [tensor.m_rr, tensor.m_tt, tensor.m_pp,
          tensor.m_rt, tensor.m_rp, tensor.m_tp]
    b = beach(mt, width=50, linewidth=1, facecolor='b',
              xy=(100, 0.5), axes=ax2)
    ax2.add_collection(b)

    outfile = os.path.join(outdir, 'waveforms_it_%d.png' % iteration)
    fig.savefig(outfile, format='png')
    plt.close(fig)


def correct_waveforms(st_data, st_synth, st_synth_grf6):
    # Create working copies of streams
    st_data_work = st_data.copy()
    st_synth_work = st_synth.copy()
    st_synth_grf6_work = st_synth_grf6.copy()

    # Calculate time shift from combined synthetic waveforms
    dt, CC = calc_timeshift(st_data_work, st_synth_work)

    # Create new stream with time-shifted synthetic seismograms
    st_synth_shift = obspy.Stream()
    for tr in st_synth_work:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        # Correct mispicked phases and set the timeshift to 0 there
        # if abs(dt[code]) > tr.times()[-1] * 0.25:
        #     dt[code] = 0.0
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

    # Fill stream with amplitude-corrected synthetic seismograms
    for tr in st_synth_shift:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        tr.data *= dA[code]

    # Fill stream with amplitude-corrected Green's functions
    for tr in st_synth_grf6_shift:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
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
        if not hasattr(stats, 'sac'):
            stat = inv.select(network=stats.network,
                              station=stats.station,
                              location=stats.location,
                              channel=stats.channel)
            print(stats.network, stats.station, stats.location, stats.channel)
            stats.sac = dict(stla=stat[0][0].latitude,
                             stlo=stat[0][0].longitude)


def create_Toeplitz(data):
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
    return linalg.toeplitz(first_col, first_row)


def create_Toeplitz_mult(stream):
    nstat = len(stream)
    npts = stream[0].stats.npts
    G = np.zeros((nstat * npts, npts))
    # print('G:')
    for istat in range(0, nstat):
        # print(istat, stream[istat].stats.station)
        G[istat * npts:(istat + 1) * npts][:] = \
            create_Toeplitz(stream[istat].data)
    return G


def create_matrix_STF_inversion(st_data, st_synth):
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
    G = create_Toeplitz_mult(st_synth)
    # GTG = np.matmul(G.transpose, G)
    # GTGmean = np.diag(GTG).mean()
    # J = np.diag(np.linspace(0, GTGmean, GTG.shape[0]))
    # A = np.matmul(np.inv(np.matmul(GTG, J))

    return d, G


def invert_STF(st_data, st_synth, method='bound_lsq'):
    print('Using %d stations for STF inversion' % len(st_data))
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

    # st_data.write('data.mseed', format='mseed')
    # st_synth.write('synth.mseed', format='mseed')

    d, G = create_matrix_STF_inversion(st_data, st_synth)

    if method == 'bound_lsq':
        m = lsq_linear(G, d, (-0.1, 1.1))
        stf = np.r_[m.x[(len(m.x) - 1) / 2:], m.x[0:(len(m.x) - 1) / 2]]
    elif method == 'lsq':
        stf, residual, rank, s = np.linalg.lstsq(G, d)
    else:
        raise ValueError('method %s unknown' % method)

    return stf


def pick(trace, threshold):
    """
    i = pick(signal, threshold)

    Return first index of signal, which crosses threshold from below.
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
    i : int
        Index of first surpassing of threshold

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


def taper_signal(trace, t_begin, t_end):
    """
    taper_signal(trace, t_begin, t_end)

    Taper data array in trace with an asymmetric Hanning window. The range
    between t_begin and t_end is left unchanged. The two seconds before t_begin
    are tapered with the rising half of a Hanning window. The (t_end-t_begin)
    seconds after t_end are tapered with the decaying half of a Hanning window.


    Parameters
    ----------
    trace : obspy.Trace
        ObsPy trace object with signal.

    t_begin : Begin of signal window

    t_end : End of signal window


    Returns
    -------
    None
        Note that the data array in the trace is modified in place.

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


def taper_before_arrival(st_data, st_synth):
    """
    taper_before_arrival(st_data, st_synth)

    Taper corresponding data in both streams before the arrival of seismic
    energy in the synthetic seismogram.


    Parameters
    ----------
    st_data : obspy.Stream
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

    for tr in st_data:
        tr_synth = st_synth.select(station=tr.stats.station,
                                   network=tr.stats.network,
                                   location=tr.stats.location)[0]
        threshold = max(abs(tr_synth.data)) * 1e-2
        arr_time = pick(tr_synth, threshold=threshold)
        taper_signal(tr, t_begin=arr_time, t_end=arr_time + 30.0)

        len_win = max(len_win, 30.0)
        arr_times['%s.%s' % (tr.stats.station, tr.stats.location)] = arr_time

    return len_win, arr_times


def main():

    helptext = 'Invert for moment tensor and source time function'
    parser = argparse.ArgumentParser(description=helptext)

    helptext = 'Path to directory with the data to use'
    parser.add_argument('--data_path', help=helptext, default='BH')

    helptext = 'Path to StationXML file'
    parser.add_argument('--event_file', help=helptext,
                        default='../EVENTS-INFO/catalog.ml')

    helptext = 'Path to Instaseis Database'
    parser.add_argument('--db_path', help=helptext,
                        default='syngine://ak135f_2s')

    helptext = 'Minimum depth (in kilometer)'
    parser.add_argument('--min_depth', help=helptext,
                        default=0.0, type=float)

    helptext = 'Maximum depth (in kilometer)'
    parser.add_argument('--max_depth', help=helptext,
                        default=20.0, type=float)

    # Parse input arguments
    args = parser.parse_args()

    # Run the main program
    for depth in np.arange(args.min_depth * 1e3,
                           args.max_depth * 1e3,
                           step=1e3, dtype=float):
        inversion(args.data_path, args.event_file, db_path=args.db_path,
                  depth_in_m=depth, dist_min=30.0, dist_max=100.0, CClim=0.6,
                  phase_list=('P', 'Pdiff'),
                  pre_offset=15,
                  post_offset=36.1,
                  work_dir='.')
