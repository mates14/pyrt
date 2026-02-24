#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.neighbors import KDTree
import astropy.wcs
import time

def find_closest_object(data, ra, dec, id_limit):
    """
    Find the closest object to the given RA/DEC within the id_limit using KDTree.

    Args:
        data: Table with ALPHA_J2000, DELTA_J2000 columns
        ra: Target right ascension in degrees
        dec: Target declination in degrees
        id_limit: Identification limit in arcseconds

    Returns:
        The closest object or None if no object is within the limit
    """
    if 'ALPHA_J2000' not in data.colnames or 'DELTA_J2000' not in data.colnames:
        return None

    # Convert arcseconds to degrees for the search radius
    search_radius_deg = id_limit / 3600.0

    # Calculate the approximate conversion factor from degrees to cartesian distance
    # at the target declination (this is a small-angle approximation)
    ra_scale = np.cos(np.radians(dec))

    # Create coordinates for data points and target
    data_coords = np.array([
        data['ALPHA_J2000'] * ra_scale,
        data['DELTA_J2000']
    ]).T

    target_coords = np.array([[ra * ra_scale, dec]])

    # Build KDTree for the data coordinates
    tree = KDTree(data_coords)

    # Find objects within the search radius
    indices = tree.query_radius(target_coords, r=search_radius_deg, count_only=False)[0]

    if len(indices) == 0:
        return None

    # If multiple objects are found, get the closest one
    if len(indices) > 1:
        # Need to find the closest among the matches
        distances = tree.query(target_coords)[0][0]
        closest_idx = indices[np.argmin(distances[indices])]
    else:
        closest_idx = indices[0]

    return data[closest_idx]

def get_pixel_scale(meta):
    """Get pixel scale in arcsec/pixel from WCS metadata."""
    # Try CD matrix first
    cd1_1 = meta.get('CD1_1', 0)
    cd1_2 = meta.get('CD1_2', 0)
    cd2_1 = meta.get('CD2_1', 0)
    cd2_2 = meta.get('CD2_2', 0)

    if cd1_1 != 0 or cd1_2 != 0:
        # Pixel scale from CD matrix (in arcsec)
        scale = np.sqrt(abs(cd1_1 * cd2_2 - cd1_2 * cd2_1)) * 3600.0
        return scale

    # Try CDELT
    cdelt1 = meta.get('CDELT1', 0)
    cdelt2 = meta.get('CDELT2', 0)
    if cdelt1 != 0:
        return abs(cdelt1) * 3600.0

    # Default fallback (typical CCD scale)
    return 0.7  # arcsec/pixel


def format_output_line(ecsv_file, meta, target, wssrndf=None):
    """Format the output line in the same style as dophot."""
    # Extract metadata
    jd = meta.get('JD', 0.0)
    chartime = jd + meta.get('EXPTIME', 0) / 2.0 / 86400.0
    filt = meta.get('FILTER', 'None')
    exptime = meta.get('EXPTIME', 0)
    airmass = meta.get('AIRMASS', 0.0)
    idnum = meta.get('IDNUM', 0)
    magzero = meta.get('MAGZERO', 0.0)
    dmagzero = meta.get('DMAGZERO', 0.0)
    wssrndf = meta.get('WSSRNDF', 0.0) if wssrndf is None else wssrndf
    limflx3 = meta.get('LIMFLX3', 0.0)
    tarid = meta.get('TARGET', 0)
    obsid = meta.get('OBSID', 0)

    if target is not None:
        # Target found
        mag = target['MAG_CALIB'] if 'MAG_CALIB' in target.colnames else target['MAG_AUTO']
        mag_err = target['MAGERR_CALIB'] if 'MAGERR_CALIB' in target.colnames else target['MAGERR_AUTO']

        out_line = f"{ecsv_file} {jd:.6f} {chartime:.6f} {filt} {exptime:3.0f} {airmass:6.3f} {idnum:4d} {magzero:7.3f} {dmagzero:6.3f} {limflx3+magzero:7.3f} {wssrndf:6.3f} {mag:7.3f} {mag_err:6.3f} {tarid} {obsid} ok"
    else:
        # Target not found
        out_line = f"{ecsv_file} {jd:.6f} {chartime:.6f} {filt} {exptime:3.0f} {airmass:6.3f} {idnum:4d} {magzero:7.3f} {dmagzero:6.3f} {limflx3+magzero:7.3f} {wssrndf:6.3f}  -       -      {tarid} {obsid} not_found"

    return out_line


def format_astrometry_line(ecsv_file, meta, target):
    """Format astrometric output line with position and quality data.

    Output columns:
    - filename
    - JD
    - RA (deg)
    - Dec (deg)
    - RA_err (arcsec) - position uncertainty in RA
    - Dec_err (arcsec) - position uncertainty in Dec
    - FWHM (arcsec)
    - SNR - signal-to-noise ratio
    - ellipticity
    - flags
    - WSSRNDF - WCS fit quality
    - mag - for reference/weighting
    - status
    """
    jd = meta.get('JD', 0.0)
    wssrndf = meta.get('WSSRNDF', 0.0)
    pixel_scale = get_pixel_scale(meta)

    if target is not None:
        # Extract position
        ra = target['ALPHA_J2000']
        dec = target['DELTA_J2000']

        # Position errors from pixel variances (convert to arcsec)
        errx2 = target.get('ERRX2_IMAGE', 0.01)
        erry2 = target.get('ERRY2_IMAGE', 0.01)
        ra_err = np.sqrt(errx2) * pixel_scale  # arcsec
        dec_err = np.sqrt(erry2) * pixel_scale  # arcsec

        # FWHM in arcsec
        fwhm_pix = target.get('FWHM_IMAGE', 3.0)
        fwhm = fwhm_pix * pixel_scale

        # SNR from magnitude error (SNR ~ 1.086/magerr for Poisson)
        mag_err = target['MAGERR_CALIB'] if 'MAGERR_CALIB' in target.colnames else target['MAGERR_AUTO']
        if mag_err > 0 and mag_err < 10:
            snr = 1.086 / mag_err
        else:
            snr = 0.0

        # Ellipticity and flags
        ellip = target.get('ELLIPTICITY', 0.0)
        flags = target.get('FLAGS', 0)

        # Magnitude for reference
        mag = target['MAG_CALIB'] if 'MAG_CALIB' in target.colnames else target['MAG_AUTO']

        out_line = (f"{ecsv_file} {jd:.6f} {ra:.7f} {dec:.7f} "
                   f"{ra_err:.4f} {dec_err:.4f} {fwhm:.3f} {snr:.1f} "
                   f"{ellip:.3f} {flags:3d} {wssrndf:.4f} {mag:.3f} ok")
    else:
        out_line = (f"{ecsv_file} {jd:.6f} - - "
                   f"- - - - "
                   f"- - {wssrndf:.4f} - not_found")

    return out_line

def process_ecsv_file(ecsv_file, ra, dec, id_limit, append_to_file=None, verbose=False, astrometry=False):
    """Process a single ECSV file and print the result.

    Args:
        ecsv_file: Path to ECSV file
        ra: Target RA in degrees
        dec: Target Dec in degrees
        id_limit: Identification limit in arcseconds
        append_to_file: Optional file to append output to
        verbose: Print timing information
        astrometry: If True, output astrometric data instead of photometry
    """
    try:
        # Start timing if verbose
        if verbose:
            start_time = time.time()

        # Read the ECSV file
        data = Table.read(ecsv_file, format='ascii.ecsv')

        if verbose:
            read_time = time.time()
            print(f"Read file in {read_time - start_time:.3f}s", file=sys.stderr)

        # Find the closest object to the target
        target = find_closest_object(data, ra, dec, id_limit)

        if verbose:
            match_time = time.time()
            print(f"Found match in {match_time - read_time:.3f}s", file=sys.stderr)

        # Format the output line (astrometric or photometric)
        if astrometry:
            output_line = format_astrometry_line(ecsv_file, data.meta, target)
        else:
            output_line = format_output_line(ecsv_file, data.meta, target)

        # Print the result
        print(output_line)

        # Append to file if requested
        if append_to_file:
            with open(append_to_file, "a") as out_file:
                out_file.write(output_line + "\n")

        if verbose:
            end_time = time.time()
            print(f"Total processing time: {end_time - start_time:.3f}s", file=sys.stderr)

        return True

    except Exception as e:
        print(f"Error processing {ecsv_file}: {e}", file=sys.stderr)
        return False

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract photometric or astrometric data for a target from ECSV files.")
    parser.add_argument("ra", type=float, help="Target right ascension in degrees")
    parser.add_argument("dec", type=float, help="Target declination in degrees")
    parser.add_argument("files", nargs="+", help="ECSV files to process")
    parser.add_argument("--id-limit", type=float, default=3.0,
                        help="Identification limit in arcseconds (default: 3.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Append output to this file in addition to stdout")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose processing information")
    parser.add_argument("--astrometry", "-a", action="store_true",
                        help="Output astrometric data (RA, Dec, position errors, FWHM, SNR) instead of photometry")
    parser.add_argument("--header", "-H", action="store_true",
                        help="Print column header as first line")

    args = parser.parse_args()

    # Print header if requested
    if args.header:
        if args.astrometry:
            print("# file JD RA Dec RA_err Dec_err FWHM SNR ellip flags WSSRNDF mag status")
        else:
            print("# file JD chartime filter exptime airmass idnum magzero dmagzero limmag wssrndf mag magerr target obsid status")

    # Initialize output file if specified
    if args.output and not os.path.exists(args.output):
        with open(args.output, "w") as f:
            pass  # Just create an empty file

    # Process files
    start_total = time.time()
    success_count = 0

    for ecsv_file in args.files:
        if os.path.isfile(ecsv_file) and ecsv_file.endswith('.ecsv'):
            if process_ecsv_file(ecsv_file, args.ra, args.dec, args.id_limit, args.output, args.verbose, args.astrometry):
                success_count += 1
        elif os.path.isfile(ecsv_file + '.ecsv'):
            # Try with .ecsv extension if not provided
            if process_ecsv_file(ecsv_file + '.ecsv', args.ra, args.dec, args.id_limit, args.output, args.verbose, args.astrometry):
                success_count += 1
        else:
            print(f"File not found: {ecsv_file}", file=sys.stderr)

    if args.verbose:
        print(f"Total runtime: {time.time() - start_total:.3f}s for {success_count} files", file=sys.stderr)

    if success_count == 0:
        print(f"No valid ECSV files were processed.", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
