#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  stfinv.py
#   Purpose:   Routines to load event information
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

import pickle
import obspy


def get_event_from_obspydmt(fnam):
    try:
        return get_event_from_pickle_file(fnam)
    except:
        return get_event_from_quake_file(fnam)


def get_event_from_pickle_file(fnam):
    quake = pickle.load(open(fnam, 'rb'), encoding='latin1')

    origin = obspy.core.event.Origin()

    origin.resource_id = str(quake['origin_id'].resource_id)
    origin.time = quake['datetime']
    origin.latitude = quake['latitude']
    origin.longitude = quake['longitude']
    origin.depth = quake['depth']

    mag = obspy.core.event.Magnitude()
    mag.mag = quake['magnitude']
    mag.magnitude_type = quake['magnitude_type']

    event = obspy.core.event.Event()
    event.magnitudes.append(mag)
    event.origins.append(origin)

    return event


def get_event_from_quake_file(fnam):
    with open(fnam) as fid:
        year, jday = [int(i) for i in fid.readline().split()]
        hour, minute, sec, microsec = [int(i) for i in fid.readline().split()]
        time = obspy.UTCDateTime(year=year, julday=jday, hour=hour,
                                 minute=minute, second=sec,
                                 microsecond=microsec)
        lat, lon = [float(i) for i in fid.readline().split()]
        depth = float(fid.readline())
        magnitude = float(fid.readline())

    origin = obspy.core.event.Origin()

    origin.time = time
    origin.latitude = lat
    origin.longitude = lon
    origin.depth = depth

    mag = obspy.core.event.Magnitude()
    mag.mag = magnitude

    event = obspy.core.event.Event()
    event.magnitudes.append(mag)
    event.origins.append(origin)

    return event
