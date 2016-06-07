import numpy as np


__all__ = ["seiscomp_to_moment_tensor"]


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
