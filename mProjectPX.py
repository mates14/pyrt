#!/usr/bin/env python3

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

# Projections that mProjectPP can handle directly
MPROJECT_PP_PROJECTIONS = {'TAN', 'SIN', 'ZEA', 'STG', 'ARC'}

def parse_args():
    """Parse command line arguments to match mProject/mProjectPP interface"""
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
    """Write status message to file if specified"""
    if status_file:
        with open(status_file, 'a') as f:
            f.write(message + '\n')
    print(message)

def check_projection_handling(header, force_convert=False):
    """Determine how to handle the projection"""
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
    """Build mProjectPP command with all relevant arguments"""
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
        # Use mProjectPP directly
        write_status(args.status, f"Using mProjectPP directly for projection {header['CTYPE1']}")
        cmd = build_mprojectpp_command(args)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            write_status(args.status, f"Error running mProjectPP: {e}")
            sys.exit(1)
            
    elif handling == 'convert':
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert to TAN+SIP
            temp_fits = os.path.join(tmpdir, 'temp_tan.fits')
#            write_status(args.status, f"Converting {header['CTYPE1']} to TAN+SIP")
            
            # Copy input file to temp location
            shutil.copy2(args.input_file, temp_fits)
            
            try:
                # Convert to TAN+SIP using the cleaner implementation
                fitter, rms = zpn_to_tan_mesh(header, ngrid=200, sip_order=args.sip_order)
                
                # Write converted file
                fitter.write(temp_fits)
                write_status(args.status, f"Converted {header['CTYPE1'][5:]} to TAN-SIP with RMS error {rms:.3f} pixels")
                
                # Build and run mProjectPP command
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
