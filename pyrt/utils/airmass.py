#!/usr/bin/env python3
"""
airmass.py

Astronomical airmass calculation utilities for pyrt
"""

import numpy as np


def rozenberg_airmass(zenith_angle_rad):
    """
    Calculate airmass using the Rozenberg (1966) formula.

    More accurate than simple sec(z) approximation, especially at high zenith angles.

    Parameters
    ----------
    zenith_angle_rad : float or array
        Zenith angle in radians

    Returns
    -------
    float or array
        Airmass value(s)

    Notes
    -----
    Formula: airmass = 1/(cos(z) + 0.025*exp(-11*cos(z)))

    References
    ----------
    Rozenberg, G.V. 1966, "Twilight: A Study in Atmospheric Optics"
    """
    cos_zenith = np.cos(zenith_angle_rad)
    airmass = 1.0 / (cos_zenith + 0.025 * np.exp(-11.0 * cos_zenith))
    return airmass


def altitude_to_airmass(altitude_rad):
    """
    Convert altitude to airmass using Rozenberg formula.

    Parameters
    ----------
    altitude_rad : float or array
        Altitude angle in radians (0 = horizon, π/2 = zenith)

    Returns
    -------
    float or array
        Airmass value(s)
    """
    zenith_angle_rad = np.pi/2.0 - altitude_rad
    return rozenberg_airmass(zenith_angle_rad)


def calculate_airmass_array(ra, dec, latitude, longitude, altitude_m, jd):
    """
    Calculate airmass for arrays of RA/Dec given observer location and time.

    This function isolates the slow astropy imports (Time, SkyCoord, AltAz,
    EarthLocation, units) so they're only loaded when actually computing airmass.

    Parameters
    ----------
    ra : array
        Right ascension in degrees
    dec : array
        Declination in degrees
    latitude : float
        Observer latitude in degrees
    longitude : float
        Observer longitude in degrees
    altitude_m : float
        Observer altitude in meters
    jd : float
        Julian date of observation

    Returns
    -------
    array
        Airmass values for each coordinate

    Notes
    -----
    Imports astropy modules only when called, not at module import time.
    This keeps the module fast to import if airmass calculation isn't needed.
    """
    # Import slow astropy modules only when needed
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, AltAz, EarthLocation
    import astropy.units as u

    # Create observer location
    loc = EarthLocation(lat=latitude*u.deg,
                       lon=longitude*u.deg,
                       height=altitude_m*u.m)

    # Create time object
    time = Time(jd, format='jd')

    # Create sky coordinates
    coords = SkyCoord(ra*u.deg, dec*u.deg)

    # Transform to altitude/azimuth
    altaz = coords.transform_to(AltAz(obstime=time, location=loc))

    # Calculate airmass using Rozenberg formula
    airmass = altitude_to_airmass(altaz.alt.rad)

    return airmass