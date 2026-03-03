"""
Astrometry utilities for high-precision position corrections.

This module provides functions for applying proper motion and parallax
corrections to catalog positions.
"""

import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
import astropy.units as u


def apply_parallax_correction(ra, dec, parallax_mas, jd):
    """Apply parallax correction to star positions.

    Corrects catalog positions for the apparent shift due to Earth's
    orbital motion around the Sun. This is necessary for ~20 mas precision
    astrometry.

    Args:
        ra: Right ascension in degrees (scalar or array)
        dec: Declination in degrees (scalar or array)
        parallax_mas: Parallax in milliarcseconds (scalar or array)
        jd: Julian date of observation

    Returns:
        tuple: (ra_corrected, dec_corrected) in degrees

    Notes:
        The parallax correction is computed using the Earth's barycentric
        position. Stars with zero or negative parallax are not corrected.

        The formula used is the standard parallactic displacement:
            Δα = π * (Y cos(α) - X sin(α)) / cos(δ)
            Δδ = π * (Z cos(δ) - X cos(α) sin(δ) - Y sin(α) sin(δ))

        where (X, Y, Z) is Earth's barycentric position in AU (ICRS), π is
        parallax in arcsec, and (α, δ) are the coordinates. Stars shift
        toward Earth's position (away from the Sun).
    """
    # Convert to arrays for uniform handling
    ra = np.atleast_1d(np.asarray(ra, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec, dtype=np.float64))
    parallax_mas = np.atleast_1d(np.asarray(parallax_mas, dtype=np.float64))

    # Get Earth's barycentric position at observation time
    t = Time(jd, format='jd')
    earth = get_body_barycentric('earth', t)

    # Earth position in AU
    X = earth.x.to(u.AU).value
    Y = earth.y.to(u.AU).value
    Z = earth.z.to(u.AU).value

    # Convert to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    # Parallax in arcsec
    parallax_arcsec = parallax_mas / 1000.0

    # Parallax displacement (in arcsec)
    # Standard formulae from spherical astronomy
    # Stars shift toward Earth's position (away from Sun)
    d_ra_arcsec = parallax_arcsec * (Y * np.cos(ra_rad) - X * np.sin(ra_rad)) / np.cos(dec_rad)
    d_dec_arcsec = parallax_arcsec * (
        Z * np.cos(dec_rad)
        - X * np.cos(ra_rad) * np.sin(dec_rad)
        - Y * np.sin(ra_rad) * np.sin(dec_rad)
    )

    # Handle zero/negative parallax (distant stars or bad measurements)
    mask = parallax_mas <= 0
    d_ra_arcsec[mask] = 0.0
    d_dec_arcsec[mask] = 0.0

    # Convert displacement to degrees and apply
    ra_corrected = ra + d_ra_arcsec / 3600.0
    dec_corrected = dec + d_dec_arcsec / 3600.0

    # Return scalar if input was scalar
    if len(ra_corrected) == 1:
        return float(ra_corrected[0]), float(dec_corrected[0])

    return ra_corrected, dec_corrected


def apply_proper_motion(ra, dec, pmra, pmdec, jd, ref_epoch=2457204.5):
    """Apply proper motion correction to star positions.

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        pmra: Proper motion in RA (deg/yr, already includes cos(dec) factor)
        pmdec: Proper motion in Dec (deg/yr)
        jd: Julian date of observation
        ref_epoch: Reference epoch as JD (default: 2457204.5 = Gaia DR2 epoch 2015.5)

    Returns:
        tuple: (ra_corrected, dec_corrected) in degrees
    """
    # Time since reference epoch in years
    dt_years = (jd - ref_epoch) / 365.2425

    ra_corrected = ra + pmra * dt_years
    dec_corrected = dec + pmdec * dt_years

    return ra_corrected, dec_corrected


def apply_astrometric_corrections(ra, dec, pmra, pmdec, parallax_mas, jd,
                                   ref_epoch=2457204.5, apply_pm=True, apply_plx=True):
    """Apply both proper motion and parallax corrections.

    This is a convenience function that applies both corrections in the
    correct order (proper motion first, then parallax).

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        pmra: Proper motion in RA (deg/yr)
        pmdec: Proper motion in Dec (deg/yr)
        parallax_mas: Parallax in milliarcseconds
        jd: Julian date of observation
        ref_epoch: Reference epoch as JD (default: Gaia DR2 2015.5)
        apply_pm: Whether to apply proper motion correction
        apply_plx: Whether to apply parallax correction

    Returns:
        tuple: (ra_corrected, dec_corrected) in degrees
    """
    ra_corr, dec_corr = ra, dec

    if apply_pm:
        ra_corr, dec_corr = apply_proper_motion(ra_corr, dec_corr, pmra, pmdec, jd, ref_epoch)

    if apply_plx:
        ra_corr, dec_corr = apply_parallax_correction(ra_corr, dec_corr, parallax_mas, jd)

    return ra_corr, dec_corr
