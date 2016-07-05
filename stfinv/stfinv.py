import numpy as np
import obspy
import os
import instaseis
from scipy import signal
import argparse
from .utils.results import Results
from .utils.depth import Depth
from .utils.iteration import Iteration
from .utils.inversion import invert_MT, invert_STF
from .utils.stream import Stream


__all__ = ["inversion",
           "correct_waveforms"]


def calc_D_misfit(CCs):
    CC = []
    for key, value in CCs.items():
        CC.append(1. - value)

    return np.mean(CC)


def read(path):
    stream = Stream()
    stream += obspy.read(path)
    return stream


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
    st = read(data_path)

    # Fill stream with station coordinates (in SAC header)
    st.get_station_coordinates()

    # Read event file
    cat = obspy.read_events(event_file)
    event = cat[0]

    db = instaseis.open_db(db_path)

    # Calculate synthetics in GRF6 format with instaseis and cut
    # time windows around the phase arrivals (out of data and GRF6 synthetics.
    # The cuts are saved on disk for the next time.
    st_data, st_synth_grf6 = st.get_synthetics(event.origins[0],
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

    # Initialize with MT from event file
    try:
        tensor = cat[0].focal_mechanisms[0].moment_tensor.tensor
    except IndexError:
        print('No moment tensor present, using explosion. Hilarity may ensue')
        tensor = obspy.core.event.Tensor(m_rr=1e20, m_tt=1e20, m_pp=1e20,
                                         m_rp=0.0, m_rt=0.0, m_tp=0.0)

    # Init with Gaussian STF with a length T:
    # log10 T propto 0.5*Magnitude
    # Scaling is such that the 5.7 Virginia event takes 5 seconds
    duration = 10 ** (0.5 * (cat[0].magnitudes[0].mag / 5.7)) * 5.0
    print('Assuming duration of %8.1f sec' % duration)
    stf = signal.gaussian(duration * 3, duration / 2 / db.info.dt)
    # stf = np.zeros(128)
    # stf[1] = 1.

    # Define butterworth filter at database corner frequency
    b, a = signal.butter(6, Wn=((1. / (db.info.dt * 2.)) /
                                (1. / 0.2)))

    # Start values to ensure one iteration
    it = 0
    misfit_reduction = 1e8
    misfit_new = 2
    res = Depth()

    while misfit_reduction > -0.1:
        # Get synthetics for current source solution
        st_synth = st_synth_grf6.calc_synthetic_from_grf6(st_data,
                                                          stf=stf,
                                                          tensor=tensor)
        st_data_work = st_data.copy()
        st_data_work, st_synth_corr, st_synth_grf6_corr, CC, dT, dA = \
            correct_waveforms(st_data_work,
                              st_synth,
                              st_synth_grf6,
                              allow_negative_CC=True)  # (it==0))

        # st_data_work.write('data.mseed', format='MSEED')
        # st_synth_grf6_corr.write('grf6.mseed', format='MSEED')

        # len_win, arr_times = taper_before_arrival(st_data_work,
        #                                           st_synth_corr)
        # len_win, arr_times = taper_before_arrival(st_synth_grf6_corr,
        #                                           st_synth_corr)

        arr_times = st_synth_corr.pick()

        nstat_used = len(st_data_work.filter_bad_waveforms(CC, CClim))

        # Calculate misfit reduction
        misfit_old = misfit_new
        misfit_new = calc_D_misfit(CC)
        misfit_reduction = (misfit_old - misfit_new) / misfit_old
        res_it = Iteration(tensor=tensor,
                           stf=stf,
                           CC=CC,
                           dA=dA,
                           dT=dT,
                           arr_times=arr_times,
                           it=it,
                           CClim=CClim,
                           depth=depth_in_m,
                           misfit=misfit_new,
                           st_data=st_data_work,
                           st_synth=st_synth_corr)

        res_it.plot(outdir=os.path.join(work_dir,
                                        'waveforms_%06dkm' % depth_in_m))

        res.append(res_it)

        print('  it: %02d, misfit: %5.3f (%8.1f pct red. %d stations)' %
              (it, misfit_new, misfit_reduction * 1e2, nstat_used))

        # Stop the inversion, if no station can be used
        if (nstat_used == 0):
            break

        # Omit the STF inversion in the first round.
        if it > 0:
            st_data_stf = st_data_work.filter_bad_waveforms(CC, CClim)

            # Do the STF inversion on downsampled data to avoid high frequency
            # noise that has to be removed later
            st_data_stf.decimate(factor=5)
            st_synth_stf = st_synth_corr.filter_bad_waveforms(CC, CClim)
            st_synth_stf.decimate(factor=5)
            stf = invert_STF(st_data_stf, st_synth_stf)

            stf = lanczos_interpolation(stf,
                                        old_start=0, old_dt=0.5,
                                        new_start=0, new_dt=0.1,
                                        new_npts=128,
                                        a=8)

        tensor = invert_MT(st_data_work.filter_bad_waveforms(CC, CClim),
                           st_synth_grf6_corr.filter_bad_waveforms(CC, CClim),
                           stf=stf,
                           outdir=os.path.join(work_dir, 'focmec'))
        it += 1

    return res


def correct_waveforms(st_data, st_synth, st_synth_grf6,
                      allow_negative_CC=False):

    print('Allowing polarity reversal: ', allow_negative_CC)

    # Create working copies of streams
    st_data_work = st_data.copy()
    st_synth_work = st_synth.copy()
    st_synth_grf6_work = st_synth_grf6.copy()

    # Calculate time shift from combined synthetic waveforms
    dt, CC = st_data_work.calc_timeshift(st_synth_work, allow_negative_CC)

    # Create new stream with time-shifted synthetic seismograms
    st_synth_work.shift_waveform(dt)

    # Create new stream with time-shifted Green's functions
    st_synth_grf6_work.shift_waveform(dt)

    # Calculate amplitude misfit
    dA = st_data_work.calc_amplitude_misfit(st_synth_work)

    # Fill stream with amplitude-corrected synthetic seismograms
    for tr in st_synth_work:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        tr.data *= dA[code]

    # Fill stream with amplitude-corrected Green's functions
    for tr in st_synth_grf6_work:
        code = '%s.%s' % (tr.stats.station, tr.stats.location)
        tr.data *= dA[code]

    return st_data_work, st_synth_work, st_synth_grf6_work, CC, dt, dA


def main():

    helptext = 'Invert for moment tensor and source time function'
    parser = argparse.ArgumentParser(description=helptext)

    helptext = 'Path to directory with the data to use'
    parser.add_argument('--data_path', help=helptext, default='BH/dis.*')

    helptext = 'Path to StationXML file'
    parser.add_argument('--event_file', help=helptext,
                        default='../EVENTS-INFO/catalog.ml')

    helptext = 'Path to Instaseis Database'
    parser.add_argument('--db_path', help=helptext,
                        default='syngine://ak135f_2s')

    helptext = 'Minimum depth (in kilometer)'
    parser.add_argument('--depth_min', help=helptext,
                        default=0.0, type=float)

    helptext = 'Maximum depth (in kilometer)'
    parser.add_argument('--depth_max', help=helptext,
                        default=20.0, type=float)

    helptext = 'Depth step width (in kilometer)'
    parser.add_argument('--depth_step', help=helptext,
                        default=1.0, type=float)

    # Parse input arguments
    args = parser.parse_args()

    # Run the main program
    result = Results()
    for depth in np.arange(args.depth_min * 1e3,
                           (args.depth_max + args.depth_step) * 1e3,
                           step=args.depth_step * 1e3, dtype=float):
        result.append(inversion(args.data_path, args.event_file,
                                db_path=args.db_path,
                                depth_in_m=depth,
                                dist_min=30.0, dist_max=100.0, CClim=0.6,
                                phase_list=('P', 'Pdiff'),
                                pre_offset=15,
                                post_offset=36.1,
                                work_dir='.'))
    best_depth = result.get_best_depth()
    print(best_depth.get_best_solution())
    print(best_depth.get_best_solution().misfit)
    best_depth.get_best_solution().plot('./best_waveform')
