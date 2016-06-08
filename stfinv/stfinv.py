import numpy as np
import obspy
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth


__all__ = ["seiscomp_to_moment_tensor", "get_synthetics"]


def seiscomp_to_moment_tensor(st_in, azimuth, scalmom=1.0):
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
    m_xx = st_in.select(channel='ZSS')[0].data * np.cos(2*azimuth) / 2. + \
           st_in.select(channel='ZEP')[0].data / 3. -                     \
           st_in.select(channel='ZDD')[0].data / 6.
    m_xx *= scalmom

    m_yy = - st_in.select(channel='ZSS')[0].data * np.cos(2*azimuth) / 2. +   \
           st_in.select(channel='ZEP')[0].data / 3. -                         \
           st_in.select(channel='ZDD')[0].data / 6.
    m_yy *= scalmom

    m_zz = st_in.select(channel='ZEP')[0].data / 3. + \
           st_in.select(channel='ZDD')[0].data / 3.
    m_zz *= scalmom

    m_xy = st_in.select(channel='ZSS')[0].data * np.sin(2*azimuth)
    m_xy *= scalmom

    m_xz = st_in.select(channel='ZDS')[0].data * np.cos(azimuth)
    m_xz *= scalmom

    m_yz = st_in.select(channel='ZDS')[0].data * np.sin(azimuth)
    m_yz *= scalmom

    return m_xx, m_yy, m_zz, m_xy, m_xz, m_yz


def get_synthetics(stream, origin, db, pre_offset=5.6, post_offset=20.0,
                   dist_min=30.0, dist_max=85.0, phase_list='P'):

    km2deg = 360.0 / (2*np.pi*6378137.0)

    model = TauPyModel(model="iasp91")

    st_data = obspy.Stream()
    st_synth = obspy.Stream()

    for tr in stream:
        tr_work = tr.copy()

        distance, azi, bazi = gps2dist_azimuth(tr.stats.sac['stla'],
                                               tr.stats.sac['stlo'],
                                               origin.latitude,
                                               origin.longitude,
                                               f=0.0)
        distance *= km2deg

        if dist_min < distance < dist_max:
            tt = model.get_travel_times(distance_in_degree=distance,
                                        source_depth_in_km=origin.depth*1e-3,
                                        phase_list=phase_list)
            travel_time = origin.time + tt[0].time

            print('%6s, %8.3f degree, %8.3f sec\n' % (tr.stats.station,
                                                      distance, travel_time))

            # Trim data around P arrival time
            tr_work.trim(starttime=travel_time - pre_offset,
                         endtime=travel_time + post_offset)

            st_data.append(tr_work)

            # Get synthetics

            # inst_rec = instaseis.Receiver(latitude=tr.stats.sac['stla'],
            #                               longitude=tr.stats.sac['stlo'],
            #                               network=tr.stats.network,
            #                               location=tr.stats.location,
            #                               station=tr.stats.station)

            gf_synth = db.get_greens_function(distance,
                                              source_depth_in_m=origin.depth,
                                              dt=0.1)
            for tr_synth in gf_synth:
                tr_synth.stats['starttime'] = tr_synth.stats.starttime + \
                                              float(origin.time)

                tr_synth.trim(starttime=travel_time - pre_offset,
                              endtime=travel_time + post_offset)

            # Convert Green's functions from seiscomp format to one per MT
            # component, which is used later in the inversion.
            m_xx, m_yy, m_zz, m_xy, m_xz, m_yz = seiscomp_to_moment_tensor(gf_synth,
                                                                           azimuth=azi,
                                                                           scalmom=1)
            data = [m_xx, m_yy, m_zz, m_xy, m_xz, m_yz]
            channels = ['MXX', 'MYY', 'MZZ', 'MXY', 'MXZ', 'MYZ']

            for icomp in range(0, 6):
                tr = obspy.Trace(data=data[icomp],
                                 header=tr_work.stats)
                tr.stats['channel'] = channels[icomp]
                st_synth.append(tr)

        else:
            print('%6s, %8.3f degree, out of range\n' %
                  (tr.stats.station, distance))

    return st_data, st_synth
