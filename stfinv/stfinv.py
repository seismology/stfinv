import numpy as np
import obspy
import glob
import os
import instaseis
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from scipy import signal, linalg
import scipy.fftpack as fft
import matplotlib.pyplot as plt


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
           "pick"]


def inversion(data_path, event_file, db_path='syngine://ak135f_2s',
              depth=-1, dist_min=30.0, dist_max=100.0,
              phase_list=('P', 'Pdiff'),
              work_dir='testinversion'):

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # Read all data
    st = obspy.read(data_path)

    # Fill stream with station coordinates (in SAC header)
    get_station_coordinates(st)

    # Read event file
    cat = obspy.read_events(event_file)
    event = cat[0]

    db = instaseis.open_db(db_path)
    st_data, st_synth_grf6 = get_synthetics(st,
                                            event.origins[0],
                                            db,
                                            depth=depth,
                                            out_dir=work_dir,
                                            pre_offset=15,
                                            post_offset=36.1,
                                            dist_min=dist_min,
                                            dist_max=dist_max,
                                            phase_list=phase_list)

    # Start values to ensure one iteration
    it = 0
    misfit_reduction = 1e8
    L2_new = 1e8

    # Initialize with MT from event file
    tensor = cat[0].focal_mechanisms[0].moment_tensor.tensor

    while misfit_reduction > 0.001:
        # Get synthetics for current source solution
        st_synth = calc_synthetic_from_grf6(st_synth_grf6,
                                            st_data,
                                            tensor)
        st_data_work = st_data.copy()
        st_data_work, st_synth_corr, \
            st_synth_grf6_corr, CC, dt, dA = correct_waveforms(st_data_work,
                                                               st_synth,
                                                               st_synth_grf6)
        nstat_used = len(st_data_work)

        plot_waveforms(st_data_work, st_synth_corr, CC, CClim=0.5,
                       outdir=os.path.join(work_dir, 'waveforms'),
                       iteration=it)

        # Calculate misfit reduction
        L2_old = L2_new
        L2_new = calc_L2_misfit(filter_bad_waveforms(st_data_work,
                                                     CC, CClim=0.5),
                                filter_bad_waveforms(st_synth_corr,
                                                     CC, CClim=0.5))
        misfit_reduction = (L2_old - L2_new) / L2_old

        tensor = invert_MT(filter_bad_waveforms(st_data_work,
                                                CC, CClim=0.5),
                           filter_bad_waveforms(st_synth_grf6_corr,
                                                CC, CClim=0.5),
                           outdir=os.path.join(work_dir, 'focmec'))
        it += 1

        print('Misfit reduction: From %e to %e, (%f pct, %d stations)' %
              (L2_old, L2_new, misfit_reduction * 1e2, nstat_used))


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
                   depth=-1,
                   pre_offset=5.6, post_offset=20.0,
                   dist_min=30.0, dist_max=85.0, phase_list='P'):

    # Use origin depth as default
    if depth == -1:
        depth = origin.depth

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
                                            grf6_directory=grf6_dir)

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
                                        source_depth_in_km=depth * 1e-3,
                                        phase_list=phase_list)
            travel_time = origin.time + tt[0].time

            # print('%6s, %8.3f degree, %8.3f sec\n' % (tr.stats.station,
            #                                           distance, travel_time))

            # Trim data around P arrival time
            tr_work.trim(starttime=travel_time - pre_offset,
                         endtime=travel_time + post_offset)

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
                                                  source_depth_in_m=depth,
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
        tr.write(os.path.join(grf6_dir, 'synth_%s.SAC' % tr.id),
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
            dA = np.sum(tr_a.data * tr_b.data) / np.sum(tr_b.data ** 2)

            # print('%s.%s: %4.2f ' %
            #      (tr_a.stats.station, tr_a.stats.location, dA))
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
        if CC[code] > CClim:
            st_filtered.append(tr)
    return st_filtered


def invert_MT(st_data, st_grf6, outdir='focmec'):
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

    outdir : String
        Path in which to write Beachballs for each iteration


    Returns
    -------
    tens_new : obspy.core.event.Tensor
        Updated moment tensor

    """

    # Remove bad waveforms
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
    # m_orig = [tens_orig.m_rr,
    #           tens_orig.m_tt,
    #           tens_orig.m_pp,
    #           tens_orig.m_rt,
    #           tens_orig.m_rp,
    #           tens_orig.m_tp]

    # m_est = [tens_new.m_rr,
    #          tens_new.m_tt,
    #          tens_new.m_pp,
    #          tens_new.m_rt,
    #          tens_new.m_rp,
    #          tens_new.m_tp]

    # fig = plt.figure(figsize=(5, 10))

    # obspy.imaging.beachball.beachball(fm=m_est, xy=(100, 0),
    #                                   fig=fig)
    # fig.savefig(os.path.join(outdir, 'bb.png'), format='png')
    return tens_new


def load_cut_files(data_directory, grf6_directory):
    # This can correctly retrieve the channel names of the grf6 synthetics
    # Load grf6 synthetics
    files_synth = glob.glob(os.path.join(grf6_directory, 'synth*'))
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


def plot_waveforms(st_data, st_synth, CC, CClim, iteration=-1,
                   outdir='./waveforms/'):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    nplots = len(st_data)
    nrows = int(np.sqrt(nplots)) + 1
    iplot = 0
    for tr in st_data:

        irow = np.mod(iplot, nrows)
        icol = np.int(iplot / nrows)

        normfac = max(np.abs(tr.data))

        yoffset = irow * 1.5
        xoffset = icol * 1.2

        xvals = np.linspace(0, 1, num=tr.stats.npts) + xoffset
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        if CC[code] > CClim:
            ls = '-'
        else:
            ls = '--'

        yvals = st_synth.select(station=tr.stats.station)[0].data / normfac
        l_s, = ax.plot(xvals,
                       yvals + yoffset,
                       color='r',
                       linestyle=ls,
                       linewidth=2)
        l_d, = ax.plot(xvals,
                       tr.data / normfac + yoffset,
                       color='k',
                       linestyle=ls,
                       linewidth=1.5)
        ax.text(xoffset + 0.05, yoffset + 0.4,
                '%s' % tr.stats.station)

        iplot += 1

    ax.legend((l_s, l_d), ('Synthetic', 'data'))
    if (iteration >= 0):
        ax.set_title('Waveform fits, iteration %d' % iteration)
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

    # Remove bad waveforms
    # st_synth_work = filter_bad_waveforms(st_synth_work, CC, CCmin)
    # st_data_work = filter_bad_waveforms(st_data_work, CC, CCmin)
    # st_synth_grf6_work = filter_bad_waveforms(st_synth_grf6_work, CC, CCmin)

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

    for istat in range(0, nstat):
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
    for istat in range(0, nstat):
        d[istat * npts:(istat + 1) * npts] = st_data[istat].data[0:npts]

    # Create G-matrix
    G = create_Toeplitz_mult(st_synth)
    return d, G


def invert_STF(st_data, st_synth):
    d, G = create_matrix_STF_inversion(st_data, st_synth)
    m, residual, rank, s = np.linalg.lstsq(G, d)
    return m


def pick(signal, threshold):
    """
    i = pick(signal, threshold)

    Return first index of signal, which crosses threshold from below.
    Note that the first crossing is returned, so if signal[0] is above threshold,
    this does not count.

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
    thresholded_data = signal > threshold
    threshold_edges = np.convolve([1, -1], thresholded_data, mode='same')
    threshold_edges[0] = 0

    return np.where(threshold_edges==1)[0][0]
