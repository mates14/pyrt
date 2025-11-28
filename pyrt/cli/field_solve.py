#!/usr/bin/python3
# Field Recognition / Plate Solving Wrapper
#
# This is a wrapper around astrometry.net's solve-field for initial field
# recognition and WCS solution. It provides sufficient precision for subsequent
# high-precision astrometric refinement by dophot's zpnfit.
#
# Smart WCS validation:
# - Automatically called by cat2det before processing
# - Removes old PV distortion coefficients (clash with modern SIP)
# - Only runs solve-field when WCS is actually broken
# - Uses .cat file from phcat for faster, more reliable solving
#
# (C) 2010, Markus Wildi, markus.wildi@one-arcsec.org
# (C) 2011-2012, Petr Kubanek, Institute of Physics <kubanek@fzu.cz>
# (C) 2024, Martin Jelínek, ASÚ AV ČR Ondřejov
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2, or (at your option)
#  any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#  Or visit http://www.gnu.org/licenses/gpl.html.
#

__author__ = 'kubanek@fzu.cz'

import os
import shutil
import string
import subprocess
import sys
import re
import time
import astropy.io.fits as pyfits
import astropy.table
import astropy.io.ascii
import astropy.wcs
import tempfile
import numpy
import math
import argparse

from pyrt.utils import dms
#from kapteyn import wcs

#ast_scales=[4,5,6,8,10,12,14]
ast_scales=[1,2,3,4] # ,5,6,8,10,12,14]


def check_wcs_needs_solving(fits_file, verbose=False):
    """Check if WCS needs to be solved or re-solved.

    Returns True if solving is needed, False if WCS is OK.

    Reasons to solve:
    1. WCS initialization fails
    2. No WCS present at all
    3. Old distortion standard (PV coefficients instead of SIP)
    """
    try:
        with pyfits.open(fits_file) as hdul:
            header = hdul[0].header

            # Check if basic WCS keywords exist
            has_wcs_keywords = all(k in header for k in ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2'])
            if not has_wcs_keywords:
                if verbose:
                    print(f"WCS keywords missing in {fits_file}")
                return True

            # Try to initialize WCS
            try:
                wcs = astropy.wcs.WCS(header)
                # Check if it has celestial axes
                if not wcs.has_celestial:
                    if verbose:
                        print(f"WCS exists but has no celestial axes in {fits_file}")
                    return True
            except Exception as e:
                if verbose:
                    print(f"WCS initialization failed for {fits_file}: {e}")
                return True

            # WCS seems OK
            return False

    except Exception as e:
        if verbose:
            print(f"Error checking WCS in {fits_file}: {e}")
        return True  # If we can't check, better to try solving


def remove_pv_coefficients(fits_file, verbose=False):
    """Remove old PV distortion coefficients from FITS header.

    PV coefficients clash with modern SIP distortion. Remove them and let
    astrometry.net validate or re-solve with clean WCS.

    Returns:
        True if PV coefficients were found and removed, False otherwise
    """
    try:
        with pyfits.open(fits_file, mode='update') as hdul:
            header = hdul[0].header

            # Find all PV keywords
            pv_keywords = [k for k in header.keys() if k.startswith('PV')]

            if pv_keywords:
                if verbose:
                    print(f"Removing {len(pv_keywords)} old PV distortion coefficients from {fits_file}")

                # Remove them
                for kw in pv_keywords:
                    del header[kw]

                hdul.flush()
                return True

        return False

    except Exception as e:
        if verbose:
            print(f"Error removing PV coefficients from {fits_file}: {e}")
        return False


def create_xyls_from_cat(cat_file, fits_file, output_xyls, verbose=False):
    """Convert a .cat file to .xyls format for astrometry.net

    The .xyls format is a FITS file with:
    - Primary HDU with AN_FILE='XYLS' marker
    - Binary table extension with X, Y coordinates and optionally FLUX

    Args:
        cat_file: Input .cat file from phcat
        fits_file: FITS image file (to get dimensions)
        output_xyls: Output .xyls filename
        verbose: Print debugging info

    Returns:
        (success: bool, width: int, height: int) tuple
    """
    try:
        # Read the catalog
        cat = astropy.io.ascii.read(cat_file, format='ecsv')

        # Check required columns
        if 'X_IMAGE' not in cat.colnames or 'Y_IMAGE' not in cat.colnames:
            if verbose:
                print(f"Warning: {cat_file} missing X_IMAGE/Y_IMAGE columns")
            return (False, 0, 0)

        # Get image dimensions
        with pyfits.open(fits_file) as hdul:
            header = hdul[0].header
            imagew = header.get('NAXIS1', 0)
            imageh = header.get('NAXIS2', 0)

        # Create minimal primary HDU with AN_FILE marker
        primary = pyfits.PrimaryHDU()
        primary.header['AN_FILE'] = ('XYLS', 'Astrometry.net file type')

        # Create binary table with X, Y, and optionally FLUX
        # Use numpy arrays with explicit dtype for double precision float (format='1D')
        x_col = pyfits.Column(name='X', format='1D',
                              array=cat['X_IMAGE'].astype(numpy.float64))
        y_col = pyfits.Column(name='Y', format='1D',
                              array=cat['Y_IMAGE'].astype(numpy.float64))

        cols = [x_col, y_col]

        # Add FLUX if available (helps solve-field prioritize bright stars)
        if 'FLUX_AUTO' in cat.colnames:
            flux_col = pyfits.Column(name='FLUX', format='1D',
                                    array=cat['FLUX_AUTO'].astype(numpy.float64))
            cols.append(flux_col)

        # Create the binary table HDU
        tbhdu = pyfits.BinTableHDU.from_columns(cols)

        # Create HDU list and write
        hdul = pyfits.HDUList([primary, tbhdu])
        hdul.writeto(output_xyls, overwrite=True)

        if verbose:
            print(f"Created {output_xyls} with {len(cat)} sources")

        return (True, imagew, imageh)

    except Exception as e:
        if verbose:
            print(f"Error creating .xyls file: {e}")
        return (False, 0, 0)


def validate_and_fix_wcs(fits_file, cat_file=None, scale=None, time_limit=15, verbose=False):
    """Validate WCS and fix it if needed.

    This function handles WCS validation and fixing with smart heuristics:
    1. Removes old PV distortion coefficients (clash with modern SIP)
    2. Checks if WCS is valid (can initialize, has celestial axes)
    3. Only runs solve-field if WCS is actually broken
    4. Uses provided .cat file for faster solving

    Args:
        fits_file: FITS file to check/fix
        cat_file: .cat file from phcat to use for solving (recommended)
        scale: Optional pixel scale hint in arcsec/pixel
        time_limit: Time limit for solve-field in seconds
        verbose: Print debugging info

    Returns:
        True if WCS is now valid, False if fixing failed
    """
    # First, check for and remove old PV coefficients (they clash with modern SIP)
    pv_removed = remove_pv_coefficients(fits_file, verbose=verbose)

    if pv_removed and verbose:
        print("Old PV coefficients removed, re-checking WCS...")

    # Check if solving is needed
    needs_solving = check_wcs_needs_solving(fits_file, verbose=verbose)

    if not needs_solving:
        if verbose:
            print(f"WCS in {fits_file} is OK, no solving needed")
        return True

    # WCS needs fixing - run solve-field
    if verbose:
        print(f"WCS in {fits_file} needs solving...")

    # Auto-detect .cat file if not provided
    if cat_file is None:
        base = os.path.splitext(fits_file)[0]
        potential_cat = base + ".cat"
        if os.path.exists(potential_cat):
            cat_file = potential_cat
            if verbose:
                print(f"Found catalog file: {cat_file}")

    # Extract scale from header if not provided
    if scale is None:
        try:
            with pyfits.open(fits_file) as hdul:
                header = hdul[0].header
                if 'CDELT1' in header:
                    scale = abs(header['CDELT1']) * 3600.0  # Convert to arcsec
                    if verbose:
                        print(f"Using scale from header: {scale:.3f} arcsec/pixel")
                elif 'CD1_1' in header:
                    scale = abs(header['CD1_1']) * 3600.0
                    if verbose:
                        print(f"Using scale from CD matrix: {scale:.3f} arcsec/pixel")
        except Exception as e:
            if verbose:
                print(f"Could not extract scale from header: {e}")

    # Run the solver
    solver = AstrometryScript(time=time_limit)

    for zoom in ast_scales:
        if verbose:
            print(f"Trying solve-field with zoom={zoom}...")
        result = solver.run(fits_file, scale=scale, zoom=zoom, replace=True, cat_file=cat_file)
        if result is not None:
            if verbose:
                print(f"Field solved successfully! RA={result[0]:.6f} Dec={result[1]:.6f}")
            return True

    # Solving failed
    if verbose:
        print(f"Failed to solve field for {fits_file}")
    return False


class WCSAxisProjection:
    def __init__(self,fkey):
        self.wcs_axis = None
        self.projection_type = None
        self.sip = False

        for x in fkey.split('-'):
            if x == 'RA' or x == 'DEC':
                self.wcs_axis = x
            elif x == 'TAN':
                self.projection_type = x
            elif x == 'SIP':
                self.sip = True
        if self.wcs_axis is None or self.projection_type is None:
            raise Exception('uknown projection type {0}'.format(fkey))

def xy2wcs(x, y, fitsh):
    """Transform XY pixel coordinates to WCS coordinates"""

    proj = wcs.Projection(fitsh)
    (ra, dec) = proj.toworld((x, y))
    return [ra, dec]

#    wcs1 = WCSAxisProjection(fitsh['CTYPE1'])
#    wcs2 = WCSAxisProjection(fitsh['CTYPE2'])
#    # retrieve CD matrix
#    cd = numpy.array([[fitsh['CD1_1'],fitsh['CD1_2']],[fitsh['CD2_1'],fitsh['CD2_2']]])
#    # subtract reference pixel
#    xy = numpy.array([x,y]) - numpy.array([fitsh['CRPIX1'],fitsh['CRPIX2']])
#    xy = numpy.dot(cd,xy)
#
#    if wcs1.wcs_axis == 'RA' and wcs2.wcs_axis == 'DEC':
#        dec = xy[1] + fitsh['CRVAL2']
#        if wcs1.projection_type == 'TAN':
#            if abs(dec) != 90:
#                xy[0] /= math.cos(math.radians(dec))
#        return [xy[0] + fitsh['CRVAL1'],dec]
#
#    if wcs1.wcs_axis == 'DEC' and wcs2.wcs_axis == 'RA':
#        dec = xy[0] + fitsh['CRVAL1']
#        if wcs2.projection_type == 'TAN':
#            if abs(dec) != 90:
#                xy[1] /= math.cos(math.radians(dec))
#        return [xy[1] + fitsh['CRVAL2'],dec]
#    raise Exception('unsuported axis combination {0} {1}'.format(wcs1.wcs_axis,wcs2.wcs_axis))

class AstrometryScript:
    def __init__(self, odir=None, scale_relative_error=0.25, astrometry_bin='/usr/bin', zoom=1.0, poly=2, time=30):
        """initialize the registration pipeline"""
        self.scale_relative_error = scale_relative_error
        self.astrometry_bin = astrometry_bin
        self.zoom = zoom
        self.poly = poly
        self.time = time
        self.odir = odir

    def run(self, fits_file, scale=None, ra=None, dec=None, replace=False, naxis1=None, naxis2=None, zoom=None, cat_file=None):
        """Run the field recognition pipeline

        Args:
            fits_file: FITS image file to solve
            scale: Pixel scale in arcsec/pixel
            ra: Initial RA guess (optional)
            dec: Initial Dec guess (optional)
            replace: If True, replace original file with solved version
            naxis1, naxis2: Image dimensions (optional)
            zoom: Downsample factor for solve-field
            cat_file: Optional .cat file from phcat.py to use for star positions
                     instead of letting solve-field detect stars
        """

        # Use context manager for temporary directory
        with tempfile.TemporaryDirectory(suffix='', prefix='field-solve.') as odir:

            infpath = os.path.join(odir, 'input.fits')
            shutil.copy(fits_file, infpath)

            # Handle catalog file if provided
            xyls_file = None
            img_width = None
            img_height = None
            if cat_file and os.path.exists(cat_file):
                xyls_file = os.path.join(odir, 'input.xyls')
                success, img_width, img_height = create_xyls_from_cat(cat_file, fits_file, xyls_file, verbose=True)
                if not success:
                    xyls_file = None
                    img_width = None
                    img_height = None
                    print(f"Warning: Could not create .xyls from {cat_file}, solve-field will detect stars")

            #solve_field=[self.astrometry_bin + '/solve-field', '-D', odir,'--no-plots', '--no-fits2fits']
            solve_field=[self.astrometry_bin + '/solve-field', '-D', odir,'--no-plots']


            if zoom is None:
                zoom = self.zoom
            
            if scale is not None:
                scale_low=scale*(1-self.scale_relative_error)
                scale_high=scale*(1+self.scale_relative_error)
                solve_field.append('-u')
                solve_field.append('app')
                solve_field.append('-L')
                solve_field.append(str(scale_low))
                solve_field.append('-H')
                solve_field.append(str(scale_high))
            
            # stupen polynomu pro fit, standardni 2 pro WF snimky nekdy nestacil
            # sergej: --order 4 --radius 1 -z 4 --scale-error 1
    #        solve_field.append('--scale-error 1')
    #        solve_field.append('--radius 1')
    #        solve_field.append('-t%d'%self.poly)

            solve_field.append('-T')
            solve_field.append('-l%d'%self.time)
            solve_field.append('-z%d'%zoom)
            solve_field.append('-y')
            solve_field.append('--uniformize')
            solve_field.append('10')

    # This produces problematic results (bad precision):
    #        if naxis1 is not None and naxis2 is not None:
    #            solve_field.append('--crpix-x')
    #            solve_field.append(str(naxis1/2))
    #            solve_field.append('--crpix-y')
    #            solve_field.append(str(naxis2/2))

    # This is virtually useless apart of speed (if it does not work when it should, it helps not):
    #        if ra is not None and dec is not None:
    #            solve_field.append('--ra')
    #            solve_field.append(str(ra))
    #            solve_field.append('--dec')
    #            solve_field.append(str(dec))
    #            solve_field.append('--radius')
    #            solve_field.append('15')

            # Always pass the FITS image (needed for WCS output)
            solve_field.append(infpath)

            # If we have a .xyls file from the catalog, also pass it
            # solve-field will use it for source positions instead of running source extraction
            if xyls_file:
                solve_field.append('--width')
                solve_field.append(str(img_width))
                solve_field.append('--height')
                solve_field.append(str(img_height))
                # Sort by FLUX (brightest first - descending order is default)
                solve_field.append('--sort-column')
                solve_field.append('FLUX')
                solve_field.append(xyls_file)

            #print(solve_field)
          
            print(solve_field)

            proc = subprocess.Popen(solve_field, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            radecline=re.compile(r'Field center: \(RA H:M:S, Dec D:M:S\) = \(([^,]*),(.*)\).')

            ret = None

            while True:
                a=proc.stdout.readline().decode("utf-8")
                if a == '':
                    break
            #    print(a)
                match=radecline.match(a)
                if match:
                    ret=[dms.parseDMS(match.group(1)), dms.parseDMS(match.group(2))]
            
            # Always overwrite the original file if .new file exists and solution was found
            new_file = os.path.join(odir, 'input.new')
            if ret is not None and os.path.exists(new_file):
                shutil.move(new_file, fits_file)
            
            # Temporary directory cleanup happens automatically via context manager
            
        return ret

def main():
    """Main entry point for field-solve command"""
    parser = argparse.ArgumentParser(
        description='Field recognition / plate solving wrapper for astrometry.net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool wraps astrometry.net's solve-field for initial field recognition.
It provides sufficient WCS precision for subsequent high-precision refinement
by dophot's zpnfit.

The tool runs in a temporary directory to avoid cluttering the working directory
with solve-field's intermediate files.

Examples:
  # Basic usage (solve-field detects stars):
  %(prog)s image.fits

  # Use existing catalog from phcat.py (recommended):
  %(prog)s image.fits --cat image.cat

  # Specify pixel scale hint:
  %(prog)s image.fits --scale 1.2

  # With both catalog and scale:
  %(prog)s image.fits --cat image.cat --scale 1.2
        """)

    parser.add_argument('fits_file', help='FITS image file to solve')
    parser.add_argument('--cat', dest='cat_file', metavar='FILE',
                        help='Use catalog from phcat.py instead of letting solve-field detect stars')
    parser.add_argument('--scale', type=float, metavar='ARCSEC',
                        help='Pixel scale hint in arcsec/pixel (helps solve-field)')
    parser.add_argument('--time', type=int, default=15, metavar='SEC',
                        help='Time limit for solve-field in seconds (default: 15)')

    args = parser.parse_args()

    # Check if FITS file exists
    if not os.path.exists(args.fits_file):
        print(f"Error: FITS file '{args.fits_file}' not found")
        sys.exit(1)

    # Check if catalog file exists (if specified)
    if args.cat_file and not os.path.exists(args.cat_file):
        print(f"Warning: Catalog file '{args.cat_file}' not found, solve-field will detect stars")
        args.cat_file = None

    a = AstrometryScript(time=args.time)

    # Try different zoom levels until one succeeds
    ret = None
    for zoom in ast_scales:
        ret = a.run(args.fits_file, scale=args.scale, zoom=zoom, replace=True, cat_file=args.cat_file)
        if ret is not None:
            print(f"Field solved successfully at zoom level {zoom}")
            print(f"Field center: RA={ret[0]:.6f} Dec={ret[1]:.6f}")
            break

    if ret is None:
        print("Field solving failed at all zoom levels")
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
