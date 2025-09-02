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