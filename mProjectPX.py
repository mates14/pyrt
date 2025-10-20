#!/usr/bin/env python3
"""
mProjectPX: Extended projection wrapper for Montage mProjectPP

This module provides a high-performance wrapper around Montage's mProjectPP
(plane-plane projection) tool, with automatic fallback for unsupported projections.

Background:
-----------
Montage's mProjectPP is orders of magnitude faster than the general mProject tool,
but only supports basic projection types (TAN, SIN, ZEA, STG, ARC). For advanced
projections like ZPN (Zenithal Polynomial), which are commonly used for precision
astrometric fitting, mProjectPP cannot be used directly.

Solution:
---------
This wrapper implements a clever workaround for unsupported projections:
1. Simulate a mesh of points across the image
2. Fit the complex projection (e.g., ZPN) to a TAN+SIP (Simple Imaging Polynomial)
3. Use the fast mProjectPP on the converted TAN+SIP image

Performance:
------------
For ZPN projections, this approach achieves approximately 100x speedup compared to
using the general mProject tool, while maintaining sub-pixel accuracy through the
SIP distortion terms.

Usage:
------
This tool is designed as a drop-in replacement for mProject/mProjectPP and is used
by the combine module to stack images. It accepts the same command-line arguments
as mProjectPP, with an additional -n/--sip-order option to control the polynomial
order used for fitting unsupported projections.

Example:
    mProjectPX.py input.fits output.fits template.hdr

    # With custom SIP order for better fitting
    mProjectPX.py input.fits output.fits template.hdr -n 4
"""

import os
import sys
import shutil
import tempfile
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import subprocess
from zpn_to_tan import zpn_to_tan_mesh

# This is to silence a particular annoying warning (MJD not present in a fits file)
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

# Projections that mProjectPP can handle directly without conversion.
# These are plane-plane projections that Montage's fast algorithm supports natively:
#   TAN - Gnomonic (tangent plane)
#   SIN - Orthographic/synthesis
#   ZEA - Zenithal equal area
#   STG - Stereographic
#   ARC - Zenithal equidistant
# All other projections (e.g., ZPN) require conversion to TAN+SIP
MPROJECT_PP_PROJECTIONS = {'TAN', 'SIN', 'ZEA', 'STG', 'ARC'}

def parse_args():
    """Parse command line arguments to match mProject/mProjectPP interface.

    Accepts all standard mProjectPP arguments plus an additional --sip-order
    option for controlling the polynomial order used in projection conversion.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='mProjectPX: Extended projection tool using TAN+SIP approximation')
    
    # Arguments matching mProject/mProjectPP
    parser.add_argument('input_file', help='Input FITS file')
    parser.add_argument('output_file', help='Output FITS file')
    parser.add_argument('template_file', help='Template header file')
    parser.add_argument('-z', '--factor', type=float, help='Pixel scale factor')
    parser.add_argument('-d', '--debug', type=int, help='Debug level')
    parser.add_argument('-s', '--status', help='Status file for output messages')
    parser.add_argument('--hdu', type=int, default=0, help='HDU to process')
    parser.add_argument('-x', '--scale', type=float, help='Pixel scale')
    parser.add_argument('-w', '--weight', help='Input weight (area) file')
    parser.add_argument('-W', '--fixed_weight', type=float, help='Fixed weight value')
    parser.add_argument('-t', '--threshold', type=float, help='Threshold for pixel values')
    parser.add_argument('-X', '--expand', action='store_true', help='Expand output image')
    parser.add_argument('-b', '--border', help='Border string specification')
    parser.add_argument('-e', '--energy', action='store_true', help='Energy mode')
    parser.add_argument('-f', '--full', action='store_true', help='Full region mode')
    parser.add_argument('-o', '--altout', help='Alternate output header')
    parser.add_argument('-i', '--altin', help='Alternate input header')

    # our argument to tweak with the sip order
    parser.add_argument('-n', '--sip-order', help='SIP order for non-PP projections fitting', default=3, type=int)
    
    return parser.parse_args()

def write_status(status_file, message):
    """Write status message to file if specified, and print to stdout.

    Args:
        status_file (str or None): Path to status file, or None to skip file output
        message (str): Status message to write
    """
    if status_file:
        with open(status_file, 'a') as f:
            f.write(message + '\n')
    print(message)

def check_projection_handling(header, force_convert=False):
    """Determine the appropriate handling strategy for the projection type.

    Checks if the projection can be handled directly by mProjectPP (passthrough)
    or needs conversion to TAN+SIP (convert). Also validates that the projection
    is supported by astropy WCS.

    Args:
        header (astropy.io.fits.Header): FITS header containing WCS information
        force_convert (bool): If True, force conversion even for supported types
                             (currently unused but reserved for future use)

    Returns:
        str: One of:
            - 'passthrough': Projection is natively supported by mProjectPP
            - 'convert': Projection needs conversion to TAN+SIP
            - 'unsupported: <error>': Projection is not supported at all
    """
    wcs = WCS(header)
    proj_type = wcs.wcs.ctype[0].split('-')[-1]

    if proj_type in MPROJECT_PP_PROJECTIONS:
        return 'passthrough'
    else:
        try:
            # Quick check if WCS can handle the projection at all
            test_pixels = np.array([[0, 0], [1, 1]])
            sky = wcs.all_pix2world(test_pixels, 0)
            _ = wcs.all_world2pix(sky, 0)
            return 'convert'
        except Exception as e:
            return f'unsupported: {str(e)}'

def build_mprojectpp_command(args, temp_input=None):
    """Build mProjectPP command with all relevant arguments.

    Constructs the command-line invocation for mProjectPP, passing through all
    arguments that were provided by the user.

    Args:
        args (argparse.Namespace): Parsed command line arguments
        temp_input (str, optional): Path to temporary input file. If provided,
                                   used instead of args.input_file (for converted
                                   projections)

    Returns:
        list: Command and arguments ready for subprocess execution
    """
    cmd = ['mProjectPP']

    # Add all optional arguments that were provided
    if args.factor: cmd.extend(['-z', str(args.factor)])
    if args.debug: cmd.extend(['-d', str(args.debug)])
    if args.status: cmd.extend(['-s', args.status])
    if args.hdu != 0: cmd.extend(['-h', str(args.hdu)])
    if args.scale: cmd.extend(['-x', str(args.scale)])
    if args.weight: cmd.extend(['-w', args.weight])
    if args.fixed_weight: cmd.extend(['-W', str(args.fixed_weight)])
    if args.threshold: cmd.extend(['-t', str(args.threshold)])
    if args.expand: cmd.append('-X')
    if args.border: cmd.extend(['-b', args.border])
    if args.energy: cmd.append('-e')
    if args.full: cmd.append('-f')
    if args.altout: cmd.extend(['-o', args.altout])
    if args.altin: cmd.extend(['-i', args.altin])

    # Add positional arguments
    cmd.append(temp_input if temp_input else args.input_file)
    cmd.append(args.output_file)
    cmd.append(args.template_file)

    return cmd

def main():
    """Main execution function implementing the projection handling logic.

    This function orchestrates the entire workflow:
    1. Parse command line arguments
    2. Read the input FITS header to determine projection type
    3. Choose handling strategy (passthrough vs convert)
    4. For passthrough: directly invoke mProjectPP
    5. For convert: create TAN+SIP approximation, then invoke mProjectPP
    6. Report errors and exit codes appropriately

    Exit codes:
        0: Success
        1: Error (file I/O, conversion, or mProjectPP execution failure)
    """
    args = parse_args()

    # Read input FITS file
    try:
        with fits.open(args.input_file) as hdul:
            header = hdul[args.hdu].header.copy()  # Make a copy to avoid modifying original
    except Exception as e:
        write_status(args.status, f"Error reading input file: {str(e)}")
        sys.exit(1)

    # Determine how to handle the projection
    handling = check_projection_handling(header)

    if handling == 'passthrough':
        # Use mProjectPP directly for supported projections (TAN, SIN, etc.)
        write_status(args.status, f"Using mProjectPP directly for projection {header['CTYPE1']}")
        cmd = build_mprojectpp_command(args)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            write_status(args.status, f"Error running mProjectPP: {e}")
            sys.exit(1)

    elif handling == 'convert':
        # Convert unsupported projections (like ZPN) to TAN+SIP approximation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert to TAN+SIP
            temp_fits = os.path.join(tmpdir, 'temp_tan.fits')
#            write_status(args.status, f"Converting {header['CTYPE1']} to TAN+SIP")

            # Copy input file to temp location
            shutil.copy2(args.input_file, temp_fits)

            try:
                # Convert to TAN+SIP using mesh-based fitting
                # ngrid=200 creates a 200x200 mesh of sample points
                # sip_order controls polynomial complexity (higher = more accurate but slower)
                fitter, rms = zpn_to_tan_mesh(header, ngrid=200, sip_order=args.sip_order)

                # Write converted file
                fitter.write(temp_fits)
                write_status(args.status, f"Converted {header['CTYPE1'][5:]} to TAN-SIP with RMS error {rms:.3f} pixels")

                # Build and run mProjectPP command on the converted image
                cmd = build_mprojectpp_command(args, temp_fits)
                subprocess.run(cmd, check=True)

            except Exception as e:
                write_status(args.status, f"Error during conversion: {str(e)}")
                sys.exit(1)
    else:
        write_status(args.status, f"Unsupported projection: {handling}")
        sys.exit(1)

if __name__ == "__main__":
    main()
