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
from pyrt.cli.zpn_to_tan import zpn_to_tan_mesh

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

def read_template_header(hdr_path):
    """Parse a Montage .hdr text file into an astropy FITS Header.

    Strips CONTINUE cards (long-string continuations copied verbatim by mGetHdr)
    before parsing — they are not needed for WCS and astropy rejects them in
    text-format headers.
    """
    with open(hdr_path) as f:
        lines = f.readlines()
    lines = [l for l in lines if not l.startswith('CONTINUE')]
    return fits.Header.fromstring(''.join(lines), sep='\n')


def write_template_hdr(fitter, naxis1, naxis2, hdr_path):
    """Write a fitted TAN+SIP WCS as a Montage-compatible .hdr file."""
    wcs = WCS(fitter.wcs())
    header = wcs.to_header(relax=True)   # relax=True emits SIP keywords
    with open(hdr_path, 'w') as f:
        f.write("SIMPLE  = T\n")
        f.write("BITPIX  = -32\n")
        f.write("NAXIS   = 2\n")
        f.write(f"NAXIS1  = {naxis1}\n")
        f.write(f"NAXIS2  = {naxis2}\n")
        for key, val in header.items():
            if key in ('WCSAXES',):
                continue
            if isinstance(val, str):
                f.write(f"{key:8s}= '{val}'\n")
            else:
                f.write(f"{key:8s}= {val}\n")
        f.write("END\n")


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

    Special handling: If old-style distortion coefficients (PV_i_j keywords) are
    present, forces conversion even for TAN projections. This is because Montage
    and astropy interpret these differently - astropy may see TAN while Montage
    sees PLA (Plate Carrée), causing mProjectPP to fail.

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

    # Check for old-style distortion coefficients (PV_i_j keywords)
    # These cause Montage and astropy to interpret projections differently
    has_pv_keywords = any(key.startswith('PV') for key in header)

    if has_pv_keywords:
        # Force conversion if old-style distortion is present
        # Even if astropy thinks it's TAN, Montage might see it as PLA
        return 'convert'

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

def build_mprojectpp_command(args, effective_input=None, effective_template=None):
    """Build mProjectPP command with all relevant arguments."""
    cmd = ['mProjectPP']

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

    cmd.append(effective_input    or args.input_file)
    cmd.append(args.output_file)
    cmd.append(effective_template or args.template_file)

    return cmd

def main():
    """Main execution function implementing the projection handling logic.

    Handles ZPN (or other unsupported) projections on both sides:
    - Input ZPN  → convert input FITS  to TAN+SIP temp file
    - Template ZPN → convert template .hdr to TAN+SIP temp .hdr

    Exit codes:
        0: Success
        1: Error (file I/O, conversion, or mProjectPP execution failure)
    """
    args = parse_args()

    # Read input FITS header
    try:
        with fits.open(args.input_file) as hdul:
            header = hdul[args.hdu].header.copy()
    except Exception as e:
        write_status(args.status, f"Error reading input file: {str(e)}")
        sys.exit(1)

    # Check template projection (the output grid)
    try:
        tmpl_header = read_template_header(args.template_file)
        tmpl_handling = check_projection_handling(tmpl_header)
    except Exception as e:
        write_status(args.status, f"Error reading template file: {str(e)}")
        sys.exit(1)

    input_handling = check_projection_handling(header)

    with tempfile.TemporaryDirectory() as tmpdir:
        effective_input    = args.input_file
        effective_template = args.template_file

        # Convert input if needed
        if input_handling == 'convert':
            temp_fits = os.path.join(tmpdir, 'input_tan.fits')
            shutil.copy2(args.input_file, temp_fits)
            try:
                fitter, rms = zpn_to_tan_mesh(header, ngrid=200, sip_order=args.sip_order)
                fitter.write(temp_fits)
                write_status(args.status,
                    f"Input: converted {header['CTYPE1'][5:]} to TAN+SIP (RMS {rms:.3f} px)")
                effective_input = temp_fits
            except Exception as e:
                write_status(args.status, f"Error converting input projection: {str(e)}")
                sys.exit(1)
        elif input_handling == 'passthrough':
            write_status(args.status, f"Input: using mProjectPP directly ({header['CTYPE1']})")
        else:
            write_status(args.status, f"Unsupported input projection: {input_handling}")
            sys.exit(1)

        # Convert template (output grid) if needed
        if tmpl_handling == 'convert':
            temp_hdr = os.path.join(tmpdir, 'template_tan.hdr')
            try:
                fitter, rms = zpn_to_tan_mesh(tmpl_header, ngrid=200, sip_order=args.sip_order)
                write_template_hdr(fitter,
                                   tmpl_header['NAXIS1'], tmpl_header['NAXIS2'],
                                   temp_hdr)
                write_status(args.status,
                    f"Template: converted {tmpl_header['CTYPE1'][5:]} to TAN+SIP (RMS {rms:.3f} px)")
                effective_template = temp_hdr
            except Exception as e:
                write_status(args.status, f"Error converting template projection: {str(e)}")
                sys.exit(1)
        elif tmpl_handling != 'passthrough':
            write_status(args.status, f"Unsupported template projection: {tmpl_handling}")
            sys.exit(1)

        cmd = build_mprojectpp_command(args, effective_input, effective_template)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            write_status(args.status, f"Error running mProjectPP: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
