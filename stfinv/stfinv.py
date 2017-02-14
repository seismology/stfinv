#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  stfinv.py
#   Purpose:   Entry point into stfinv
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

import numpy as np
import obspy
import os
import instaseis
from scipy import signal
import argparse
from .utils.results import Results, load_results
from .utils.stream import read
from .utils.io import get_event_from_obspydmt
from .utils.inversion import inversion

__all__ = ["inversion",
           "load_results"]


def open_files(data_path, event_file, db_path):
    # Read all data
    st = read(data_path)

    # Fill stream with station coordinates (in SAC header)
    st.get_station_coordinates()

    # Read event file
    try:
        cat = obspy.read_events(event_file)
        if len(cat) > 1:
            msg = 'File %s contains more than one event. Dont know, which one\
                   to chose. Please provide QuakeML file with just one event.'
            raise TypeError(msg)

        event = cat[0]
    except:
        event = get_event_from_obspydmt(event_file)

    origin = event.origins[0]

    db = instaseis.open_db(db_path)

    # Initialize with MT from event file
    try:
        tensor = event.focal_mechanisms[0].moment_tensor.tensor
    except IndexError:
        print('No moment tensor present, using explosion. Hilarity may ensue')
        tensor = obspy.core.event.Tensor(m_rr=1e20, m_tt=1e20, m_pp=1e20,
                                         m_rp=0.0, m_rt=0.0, m_tp=0.0)

    # Init with Gaussian STF with a length T:
    # log10 T propto 0.5*Magnitude
    # Scaling is such that the 5.7 Virginia event takes 5 seconds
    if len(event.magnitudes) > 0:
        duration = 10 ** (0.5 * (event.magnitudes[0].mag / 5.7)) * 5.0 / 2
    else:
        duration = 2.5

    print('Assuming duration of %8.1f sec' % duration)
    stf = signal.gaussian(duration * 2, duration / 4 / db.info.dt)

    return db, st, origin, tensor, stf


def define_arguments():
    helptext = 'Invert for moment tensor and source time function'
    parser = argparse.ArgumentParser(description=helptext)

    helptext = 'Path to directory with the data to use'
    parser.add_argument('--data_path', help=helptext, default='BH/dis.*')

    helptext = 'Path to StationXML file'
    parser.add_argument('--event_file', help=helptext,
                        default='./info/event.pkl')

    helptext = 'Path to Working Directory. Default is current directory.'
    parser.add_argument('--work_dir', help=helptext,
                        default='.')

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

    helptext = 'Mimimum value of CC in which Seismogram is used for inversion'
    parser.add_argument('--CClim', help=helptext,
                        default=0.6, type=float)

    helptext = 'Misfit critertion. Allowed values are CC and L2-norm'
    parser.add_argument('--misfit', help=helptext,
                        default='CC', type=str)

    helptext = 'Tolerance parameter. The algorithm terminates if the ' + \
        'relative change of the cost function is less than `tol`' + \
        'on the last iteration.'
    parser.add_argument('--tol', help=helptext,
                        default=1e-3, type=float)

    return parser


def main():

    parser = define_arguments()

    # Parse input arguments
    args = parser.parse_args()

    # Open files
    db, st, tensor, origin, stf = open_files(args.data_path,
                                             args.event_file,
                                             args.db_path)

    # Run the main program
    result = Results()
    for depth in np.arange(args.depth_min * 1e3,
                           (args.depth_max + args.depth_step) * 1e3,
                           step=args.depth_step * 1e3, dtype=float):
        tensor_init = tensor.copy()
        stf_init = stf.copy()
        result.append(inversion(db, st, tensor_init, origin, stf_init,
                                depth_in_m=depth,
                                dist_min=30.0, dist_max=100.0,
                                CClim=args.CClim,
                                phase_list=('P', 'Pdiff'),
                                pre_offset=15,
                                post_offset=60.0,  # 36.1,
                                tol=args.tol,
                                misfit=args.misfit,
                                work_dir=args.work_dir))

    # Print some info about best solution
    best_depth = result.get_best_depth()
    best_solution = best_depth.get_best_solution()
    t = best_solution.tensor

    print('')
    print('******************************************************************')
    print('Best-fitting result:')
    print('  depth: %6.2f km' % best_solution.depth)
    print('  iteration: %d' % best_solution.it)
    print('  misfit: %5.3f' % best_solution.misfit)
    print('  moment tensor: (%0.2e, %0.2e, %0.2e, %0.2e, %0.2e, %0.2e)' %
          (t.m_tt, t.m_pp, t.m_rr, t.m_tp, t.m_rt, t.m_rp))
    print('******************************************************************')
    print('')

    plotdir_best = os.path.join(args.work_dir, 'best_waveform')
    best_depth.get_best_solution().plot(plotdir_best)

    save_fnam = os.path.join(args.work_dir,
                             'result.npz')
    result.save(fnam=save_fnam)
