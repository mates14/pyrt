#!/usr/bin/env python3
""" convert a sextractor catalog + fits image file into a .det (ecsv) input for dophot
Takes the image metadata, cleans it up, and into the data body of the ecsv loads the detections.
So that dophot works with one file, can have minimal code and assume existence of certain essential keywords.
"""

import os
import sys
import argparse
from contextlib import suppress

import numpy as np
import scipy.optimize as opt

import astropy
import astropy.wcs
import astropy.table
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from pyrt.utils.file_utils import try_sex, try_ecsv, try_img
from pyrt.cli.field_solve import validate_and_fix_wcs

from typing import Optional, Tuple

def delin(number):
    """cauchy delinearization to give outliers less weight and have more robust fitting"""
    try:
        # arctan version
        # ret = np.sqrt(np.arctan(number**2))
        # cauchy version
        ret = np.sqrt(np.log1p(number**2))
    except RuntimeWarning:
        ret = np.sqrt(np.log1p(number**2))
    return ret

def errormodel(params, data):
    """residuals used when fitting the magnitude limit"""
    return delin(data[1]-(data[0]-params[0])/params[1])

def load_chip_id(header):
    """standard chip ID to be always the same"""
    try:
        chip = header['CCD_SER']
    except KeyError:
        #return -1,"unknown"
        return "unknown"

    if chip == "":
        try:
            chip = header['CCD_TYPE']
        except KeyError:
            chip = "unknown"

    if chip == "":
        chip = "unknown"

    return chip

def read_options(args=sys.argv[1:]):
    """... take over the world, what else can a readOptions method do?"""
    parser = argparse.ArgumentParser(description="Compute photometric calibration for a FITS image.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Print debugging info.")
    parser.add_argument("-o", "--output", action='store_true', help="Output file.")
    parser.add_argument("-f", "--filter", help="Override filter info from fits", type=str)
    parser.add_argument("--target-photometry", action='store_true', default=True,
                        help="Enable direct photometry of RTS2 target coordinates (default: True)")
    parser.add_argument("--force-regen", action='store_true', help="Force regeneration by running phcat even if .cat exists")
    parser.add_argument("files", help="Frames to process", nargs='+', action='extend', type=str)
    opts = parser.parse_args(args)
    return opts

def fix_time(hdr, target_photometry=True, verbose=False):
    """fixes time-related problems in a FITS header"""

    # these three need to be put in sync together with a few lower priority keywords
    jd_ctime = np.nan # CTIME made from JD
    do_ctime = np.nan # CTIME made from DATE-OBS
    ux_ctime = np.nan # Unix
    mjd_ctime = np.nan # MJD

    # This keyword is mandatory in an astronomical FITS image header
    with suppress(KeyError,ValueError): do_ctime = Time(hdr['DATE-OBS']).to_value('unix')
    if verbose and np.isnan(do_ctime): print("No DATE-OBS? That's BAD!")

    with suppress(KeyError): jd_ctime = (hdr['JD_START'] - 2440587.5) * 86400
    with suppress(KeyError): jd_ctime = (hdr['JD'] - 2440587.5) * 86400
    with suppress(KeyError): mjd_ctime = (hdr['MJD'] - 40587.0) * 86400
    with suppress(KeyError): mjd_ctime = (hdr['MJD-OBS'] - 40587.0) * 86400
    usec = 0
    with suppress(KeyError): usec = hdr['USEC']
    with suppress(KeyError): ux_ctime = np.float64(hdr['CTIME'] + usec/1000000.)

    # and now... there are up to four possible sources of time, so lets check if they are in sync
    # and sync them if they are not

    # in the older frames from RTS2, the CTIME is rounded to full minute or so, so it is not authoritative
    # most precise there is CTIME+USEC/1e6

    ux_jd = ux_ctime - jd_ctime
    ux_mjd = ux_ctime - mjd_ctime
    ux_do = ux_ctime - do_ctime
    jd_mjd = jd_ctime - mjd_ctime
    jd_do = jd_ctime - do_ctime
    mjd_do = mjd_ctime - do_ctime

    if not np.isnan(ux_jd ) and np.abs( ux_jd) > 0.001 and verbose: print(f"ux/jd  time info differ by {ux_jd:.3f}s!")
    if not np.isnan(ux_mjd) and np.abs(ux_mjd) > 0.001 and verbose: print(f"ux/mjd time info differ by {ux_mjd:.3f}s!")
    if not np.isnan(ux_do ) and  np.abs(ux_do) > 0.001 and verbose: print(f"ux/do  time info differ by {ux_do:.3f}s!")
    if not np.isnan(jd_mjd) and np.abs(jd_mjd) > 0.001 and verbose: print(f"jd/mjd time info differ by {jd_mjd:.3f}s!")
    if not np.isnan(jd_do ) and  np.abs(jd_do) > 0.001 and verbose: print(f"jd/do  time info differ by {jd_do:.3f}s!")
    if not np.isnan(mjd_do) and np.abs(mjd_do) > 0.001 and verbose: print(f"mjd/do time info differ by {mjd_do:.3f}s!")

    ctime = np.nan
    if not np.isnan(ux_ctime) and np.isnan(ctime):
        if verbose: print(f"CTIME set based on CTIME+USEC to CTIME={ux_ctime:.6f}")
        ctime = ux_ctime
    if not np.isnan(do_ctime) and np.isnan(ctime):
        if verbose: print(f"CTIME set based on DATE-OBS to CTIME={do_ctime:.6f}")
        ctime = do_ctime
    if not np.isnan(mjd_ctime) and np.isnan(ctime):
        if verbose: print(f"CTIME set based on MJD-OBS to CTIME={do_ctime:.6f}")
        ctime = mjd_ctime
    if not np.isnan(jd_ctime) and np.isnan(ctime):
        if verbose: print(f"CTIME set based on the JD to CTIME={do_ctime:.6f}")
        ctime = jd_ctime

    # old comment: this is a wrong but by far the most reliable solution (wrong:leap secs?)
    # fact: actually JD and CTIME are the same thing defined precisely this way
    # Actually, the only time dophot uses is CTIME, so I wiped from here all other time keyword messing
    if not np.isnan(ctime):
        hdr['JD'] = 2440587.5 + ctime/86400.
        if verbose: print(f"JD set based on the CTIME to JD={hdr['JD']:.6f}")

    # if I do not set this, the following WCS load will complain
    hdr['MJD-OBS'] = hdr['JD'] - 2400000.5

    time = astropy.time.Time(hdr['JD'], format='jd')

    # for stellar physics with high time resolution, BJD is very useful
    if target_photometry:
        ondrejov = EarthLocation(lat=hdr['LATITUDE']*u.deg, lon=hdr['LONGITUD']*u.deg, height=hdr['ALTITUDE']*u.m)
        with suppress(KeyError):
            target = SkyCoord(hdr['ORIRA'], hdr['ORIDEC'], unit=u.deg)
            hdr['BJD'] = hdr['JD'] + time.light_travel_time(target, kind='barycentric', location=ondrejov,
                ephemeris='builtin').to_value('jd',subfmt='float')

def calculate_background_stats(data):
    """Calculate background sigma using row differences method"""
    if data is None or len(data) < 2:
        raise ValueError("Invalid data array for background calculation")

    ndiff = np.zeros(len(data), dtype=np.float64)
    i, j = 0, 0

    while i < len(data) - 1:
        diff = abs(data[i].astype(np.float32) - data[i+1].astype(np.float32))
        median = np.nanmedian(diff)
        if not np.isnan(median):
            ndiff[j] = median
            j += 1
        i += 1

    if j == 0:
        raise ValueError("No valid background measurements")

    scale_factor = 1.0489  # scale factor of median of two point's distance to standard deviation
    sigma = np.nanmedian(ndiff[:j])
    median = np.nanmedian(data[~np.isnan(data)])

    return scale_factor * sigma, median

def get_limits(det, verbose=False):
    """ compute a detection limit from errorbars
        get all the objects from the input file
        fit it with:
        error = 10**((x-L1)/2.5)
        error = 10**((x-L1)/Q)
        guess of L1 may be the zeropoint-5, but a practical use shows a unity is good enough
        fit [16:] 10**(-(L1-x)/2.5) "d50-n2.fits.xat" u ($4+22.881):($5*log(2.5)*3):5 via L1
        and then s-sigma limit is:
        l(s)=L1+Q*log10(1.091/s)
        stability testing of these limits tested at a series of short exposures of AD Leo was 1-sigma=0.013 mag.
        level of accuracy of limits set this way depends on the underlying detection alorithm, its gain/rn settings and aperture
    """
    # Check if we have enough detections - warn but continue without setting limits
    if len(det) < 3:
        if verbose:
            print(f"Warning: Only {len(det)} detection(s), cannot fit error model")
        # Don't set LIMFLX3/LIMFLX10/DLIMFLX3 - downstream code can check if they exist
        return

    # Check if magnitude errors produce finite values
    log_errors = np.log10(det['MAGERR_AUTO'] * np.log(2.5) * 3)
    if not np.all(np.isfinite(log_errors)):
        if verbose:
            print(f"Warning: Non-finite magnitude errors, cannot fit error model")
        # Don't set LIMFLX3/LIMFLX10/DLIMFLX3 - downstream code can check if they exist
        return

    try:
        res = opt.least_squares(errormodel, [1, 2.5], args=[(det['MAG_AUTO'], log_errors)])
        det.meta['LIMFLX3'] = res.x[0]
        det.meta['LIMFLX10'] = res.x[0]+res.x[1]*np.log10(1.091/10) # FIXME: this is strange, I do not trust this is actually 10-sigma

        try:
            cov = np.linalg.inv(res.jac.T.dot(res.jac))
            fiterrors = np.sqrt(np.diagonal(cov))
            if not np.isnan(fiterrors[0]):
                det.meta['DLIMFLX3'] = fiterrors[0]
            else:
                det.meta['DLIMFLX3'] = 0
        except:
            fiterrors = res.x*np.nan
            det.meta['DLIMFLX3'] = 0

        if verbose and res.success:
            print(f"Limits fitted, LIMFLX3 = {det.meta['LIMFLX3']:.3f} +/- {det.meta['DLIMFLX3']:.3f}")
        if verbose and not res.success:
            print("Fitting limits failed")
    except (ValueError, RuntimeError) as e:
        if verbose:
            print(f"Warning: Error model fitting failed: {e}")
        # Don't set LIMFLX3/LIMFLX10/DLIMFLX3 - downstream code can check if they exist

def open_files(arg, verbose=False, force_regen=False):
    """sort out the argument and open appropriate files"""
    # either the argument may be a fits file or it may be an output of sextractor
    detf, det = try_sex(arg, verbose)
    if det is None: detf, det = try_ecsv(arg,verbose)

    imgf, img = try_img(arg, verbose)

    if det is None and img is None:
        print(f"Error: Argument {arg} is not a fits nor a sextractor catalog")
        raise FileNotFoundError

    if img is None: imgf, img = try_img(os.path.splitext(arg)[0], verbose)
    if img is None: imgf, img = try_img(os.path.splitext(arg)[0] + ".fits", verbose)
    if det is None: detf, det = try_sex(arg + ".xat", verbose)
    if det is None: detf, det = try_sex(os.path.splitext(arg)[0] + ".cat", verbose)
    if det is None:
        detf, det = try_ecsv(os.path.splitext(arg)[0] + ".cat", verbose)

    # Force regeneration if requested - run phcat even if catalog exists
    if force_regen and det is not None:
        if verbose: print(f"Force regeneration requested, will run phcat to regenerate catalog")
        det = None  # Force phcat to run

    if det is None: # os.system does not raise an exception if it fails
        # Try installed command first, fall back to module for source runs
        import shutil
        if shutil.which("pyrt-phcat"):
            cmd = f"pyrt-phcat {arg}"
        else:
            # Fall back to running as module (for source/development)
            cmd = f"{sys.executable} -m pyrt.cli.phcat {arg}"

        try:
            if verbose: print(f"Running {cmd}")
            os.system(cmd)
        except:
            print(f"Warning: Tried and failed to execute {cmd}")

        detf, det = try_ecsv(os.path.splitext(arg)[0] + ".cat",verbose)
        if det is None: detf, det = try_sex(arg + ".xat", verbose)

    if det is None or img is None:
        print(f"Error: Cannot couple fits image and sextractor list for argument {arg}")
        raise FileNotFoundError

    if os.path.getmtime(imgf) > os.path.getmtime(detf):
        if verbose: print(f"Warning: {imgf} is {os.path.getmtime(imgf)-os.path.getmtime(detf):.0f}s newer than {detf}!")

    return det,imgf,img

def fix_latlon(hdr, verbose=False):
    """ Fix the lat/lon/alt information in the file, fallback to Ondrejov if it is not there
    Ondrejov fallback is generally harmless, but the keywords are required, so it is there"""

    latitude = 49.9090806 # BART Latitude
    with suppress(KeyError): latitude = np.float64(hdr['TEL_LAT'])
    with suppress(KeyError): latitude = np.float64(hdr['LATITUDE'])
    hdr['LATITUDE'] = latitude

    longitude = 14.7819092 # BART Longitude
    with suppress(KeyError): longitude = np.float64(hdr['TEL_LONG'])
    with suppress(KeyError): longitude = np.float64(hdr['LONGITUD'])
    hdr['LONGITUD'] = longitude

    altitude = 530 # BART Altitude
    with suppress(KeyError): altitude = np.float64(hdr['TEL_ALT'])
    with suppress(KeyError): altitude = np.float64(hdr['ALTITUDE'])
    hdr['ALTITUDE'] = altitude

    if verbose: print(f"Observatory at {hdr['LONGITUD']:.3f},{hdr['LATITUDE']:.3f} at {hdr['ALTITUDE']:.0f}m ")

def fix_filter(hdr, verbose=False, opt_filter=None):
    """Fix the filter keyword (multiple variants of the same filter etc.)"""

    old_fltr = "(not_present)"
    if opt_filter is not None:
        fits_fltr = opt_filter
    else:
        try:
            old_fltr = hdr['FILTER']
            fits_fltr = old_fltr
        except KeyError:
            fits_fltr = 'N'

        if fits_fltr == "OIII": fits_fltr = "oiii"
        if fits_fltr == "Halpha": fits_fltr = "halpha"
        if fits_fltr == "u": fits_fltr = "Sloan_u"
        if fits_fltr == "g": fits_fltr = "Sloan_g"
        if fits_fltr == "r": fits_fltr = "Sloan_r"
        if fits_fltr == "i": fits_fltr = "Sloan_i"
        if fits_fltr == "z": fits_fltr = "Sloan_z"
        if fits_fltr == "UKIRT Z": fits_fltr = "UKIRT_z"
        if fits_fltr == "SDSS g\'": fits_fltr = "Sloan_g"
        if fits_fltr == "SDSS r\'": fits_fltr = "Sloan_r"
        if fits_fltr == "SDSS i\'": fits_fltr = "Sloan_i"
        if fits_fltr == "UNK": fits_fltr = "N"
        if fits_fltr == "C": fits_fltr = "N"
        if fits_fltr == "clear": fits_fltr = "N"
        if fits_fltr == "Clear": fits_fltr = "N"

    hdr['FILTER'] = fits_fltr
    if verbose: print(f"Filter changed from {old_fltr} to {fits_fltr}")

def remove_junk(hdr):
    """FITS files tend to be flooded with junk that accumulates over time
    and it is difficult to deal with"""
    for delme in ['comments','COMMENTS','history','HISTORY']:
        try:
            del hdr[delme]
        except KeyError:
            pass

def process_detections(det: astropy.table.Table,
                      fits_path: str,
                      cat_path: Optional[str] = None,
                      filter_override: Optional[str] = None,
                      target_photometry: bool = True,
                      verbose: bool = False) -> astropy.table.Table:
    """Main detection processing function that can be called programmatically

    Args:
        det: Detection catalog table
        fits_path: Path to FITS image file
        cat_path: Path to .cat catalog file (used for WCS solving if needed)
        filter_override: Override filter from command line
        target_photometry: Enable target photometry
        verbose: Print debugging info
    """

    det.meta['FITSFILE'] = fits_path

    # Validate and fix WCS if needed (before reading header)
    # This checks if WCS is valid and runs solve-field if necessary
    # Use the .cat file we already loaded for faster solving
    wcs_ok = validate_and_fix_wcs(fits_path, cat_file=cat_path, verbose=verbose)
    if not wcs_ok:
        print(f"Warning: WCS validation/fixing failed for {fits_path}, proceeding anyway...")

    # remove zeros in the error column
    det['MAGERR_AUTO'] = np.sqrt(det['MAGERR_AUTO']*det['MAGERR_AUTO']+0.0005*0.0005)
    with astropy.io.fits.open(fits_path) as fitsfile:
        img_sigma, img_median = calculate_background_stats(fitsfile[0].data)
        det.meta['MEDIAN'] = img_median
        det.meta['BGSIGMA'] = img_sigma
        c = fitsfile[0].header

    # copy most of the sensible keywords from img to det
    for i,j in c.items():
        if i in ('NAXIS', 'NAXIS1', 'NAXIS2', 'BITPIX','FWHM'): continue
        if len(i)>8: continue
        if "." in i: continue
        det.meta[i] = j

    # FIXME: this has been improved somewhere else
    det.meta['CHIP_ID'] = load_chip_id(c)

    fix_latlon(det.meta, verbose=verbose)
    fix_time(det.meta, target_photometry=target_photometry, verbose=verbose)
    get_limits(det, verbose=verbose)

    imgwcs = astropy.wcs.WCS(det.meta)

    fix_filter(det.meta, verbose=verbose, opt_filter=filter_override)

    det.meta['AIRMASS'] = 1.0
    with suppress(KeyError): det.meta['AIRMASS'] = c['AIRMASS']

    exptime = 0
    with suppress(KeyError): exptime = c['EXPTIME']
    with suppress(KeyError): exptime = c['EXPOSURE']
    det.meta['EXPTIME'] = exptime

    obsid_d = 0
    obsid_n = 0
    with suppress(KeyError): obsid_n = c['OBSID']
    with suppress(KeyError): obsid_d = c['SCRIPREP']
    det.meta['OBSID'] = f"{obsid_n:0d}.{obsid_d:02d}"

    # RTS2 true target coordinates: ORIRA/ORIDEC
    # it sounds stupid, but comes from complicated telescope pointing logic
    if target_photometry:
        det.meta['OBJRA'] = -100
        with suppress(KeyError): det.meta['OBJRA'] = np.float64(c['ORIRA'])
        det.meta['OBJDEC'] = -100
        with suppress(KeyError): det.meta['OBJDEC'] = np.float64(c['ORIDEC'])

        try:
            tmp = imgwcs.all_world2pix(det.meta['OBJRA'], det.meta['OBJDEC'], 0)
        except RuntimeError:
            print("Error: Bad astrometry, cannot continue")
            return None

        if verbose:
            with suppress(KeyError):
                print(f"Target is {det.meta['OBJECT']} at ra:{det.meta['OBJRA']} dec:{det.meta['OBJDEC']} -> x:{tmp[0]} y:{tmp[1]}")
    else:
        det.meta['OBJRA'] = -100
        det.meta['OBJDEC'] = -100

    det.meta['CTRX'] = c['NAXIS1']/2
    det.meta['CTRY'] = c['NAXIS2']/2
    # NAXIS1/2 are bound to the image - they will not stay in the header
    det.meta['IMGAXIS1'] = c['NAXIS1']
    det.meta['IMGAXIS2'] = c['NAXIS2']

    # Check if WCS has celestial information and is valid
    if not imgwcs.has_celestial:
        print("Error: No valid WCS/astrometry found in FITS header, cannot continue")
        return None

    try:
        rd = imgwcs.all_pix2world([[det.meta['CTRX'], det.meta['CTRY']], [0, 0], [det.meta['CTRX'], det.meta['CTRY']+1]], 0)

        # Check if declination is within valid range
        if abs(rd[0][1]) > 90:
            print(f"Error: Invalid WCS transformation (dec={rd[0][1]:.1f}°), likely bad SIP solution. Cannot continue")
            return None

        det.meta['CTRRA'] = rd[0][0]
        det.meta['CTRDEC'] = rd[0][1]

        center = SkyCoord(rd[0][0]*astropy.units.deg, rd[0][1]*astropy.units.deg, frame='fk5')
        corner = SkyCoord(rd[1][0]*astropy.units.deg, rd[1][1]*astropy.units.deg, frame='fk5')
        pixel = SkyCoord(rd[2][0]*astropy.units.deg, rd[2][1]*astropy.units.deg, frame='fk5').separation(center)
        field = center.separation(corner)
        if verbose:
            if field.deg > 10:
                print(f"Diagonal field size: {field.deg*2:.1f}°, Pixel size: {pixel.arcsec:.3f}\"")
            elif field.deg > 1:
                print(f"Diagonal field size: {field.deg*2:.2f}°, Pixel size: {pixel.arcsec:.3f}\"")
            else:
                print(f"Diagonal field size: {field.deg*120:.1f}\', Pixel size: {pixel.arcsec:.3f}\"")

        det.meta['FIELD'] = field.deg
        det.meta['PIXEL'] = pixel.arcsec
    except (RuntimeError, ValueError) as e:
        print(f"Error: Bad astrometry/WCS transformation ({e}), cannot continue")
        return None

    try:  # Normally we should have a FWHM value from phcat in the photometry file
        fwhm = det.meta['FWHM']
    except: # if not try some alternatives
        print(f"FWHM not found in .cat file, that's strange! Trying something else.")
        try:
            fwhm = c['FWHM']
            if verbose:
                print(f"Detected FWHM {fwhm:.1f} pixels ({fwhm*pixel.arcsec:.1f}\")")
        except KeyError:
            fwhm = 2
            if verbose:
                print(f"Set FWHM {fwhm:.1f} pixels ({fwhm*pixel.arcsec:.1f}\")")
        det.meta['FWHM'] = fwhm

    remove_junk(det.meta)

    return det

def main():
    """CLI entry point: C-like main() routine"""
    options = read_options(sys.argv[1:])

    for arg in options.files:

        try:
            det,filef,fitsfile = open_files(arg, verbose=options.verbose, force_regen=options.force_regen)
        except FileNotFoundError:
            continue

        if det is not None and 'keywords' in det.meta:
            del det.meta['keywords']

        # Derive catalog path from image path
        cat_path = os.path.splitext(filef)[0] + ".cat"

        det = process_detections(det, filef,
                               cat_path=cat_path,
                               filter_override=options.filter,
                               target_photometry=options.target_photometry,
                               verbose=options.verbose)

        if det is None:
            print("Skipping file due to errors")
            continue

        output = os.path.splitext(filef)[0] + ".det"
        if options.verbose: print(f"Writing output file {output}")
        det.write(output, format="ascii.ecsv", overwrite=True)

# this way, the variables local to main() are not globally available, avoiding some programming errors
if __name__ == "__main__":
    main()
