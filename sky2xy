#!/usr/bin/env python3
"""
wcsconv.py - Unified WCS coordinate converter
Python replacement for WCSTools sky2xy and xy2sky with enhanced functionality

Usage: Link or copy to 'sky2xy' and 'xy2sky' for automatic mode detection
Author: Python rewrite combining Doug Mink's and Jessica Mink's original C code
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import numpy as np
    from astropy.io import fits
    from astropy import wcs
    from astropy.coordinates import SkyCoord
    from astropy import units as u
except ImportError as e:
    print(f"Error: Required packages missing. Install with: pip install astropy numpy")
    print(f"Missing: {e}")
    sys.exit(1)


def detect_mode():
    """Detect conversion mode from program name."""
    prog_name = Path(sys.argv[0]).name.lower()
    if 'sky2xy' in prog_name:
        return 'sky2xy'
    elif 'xy2sky' in prog_name:
        return 'xy2sky'
    else:
        return None


def parse_coordinate(coord_str):
    """Parse coordinate string flexibly - supports degrees, sexagesimal, etc."""
    coord_str = coord_str.strip()
    
    # Try direct float conversion first
    try:
        return float(coord_str)
    except ValueError:
        pass
    
    # Handle sexagesimal formats
    # RA: hh:mm:ss.s or hh mm ss.s or hhmmss.s
    # Dec: dd:mm:ss.s or dd mm ss.s or ddmmss.s
    
    # Remove common separators and convert to standardized format
    coord_str = re.sub(r'[hmdHMD]', ' ', coord_str)  # Remove h,m,d,s markers
    coord_str = re.sub(r'[:;]', ' ', coord_str)      # Convert : and ; to spaces
    coord_str = re.sub(r'\s+', ' ', coord_str)       # Normalize whitespace
    
    parts = coord_str.split()
    
    if len(parts) == 1:
        # Single number - assume degrees
        return float(parts[0])
    elif len(parts) == 2:
        # Degrees and minutes
        deg = float(parts[0])
        min_val = float(parts[1])
        sign = -1 if deg < 0 or parts[0].startswith('-') else 1
        return sign * (abs(deg) + min_val / 60.0)
    elif len(parts) >= 3:
        # Degrees, minutes, seconds
        deg = float(parts[0])
        min_val = float(parts[1])
        sec = float(parts[2])
        sign = -1 if deg < 0 or parts[0].startswith('-') else 1
        return sign * (abs(deg) + min_val / 60.0 + sec / 3600.0)
    else:
        raise ValueError(f"Cannot parse coordinate: {coord_str}")


def parse_ra_dec(ra_str, dec_str):
    """Parse RA and Dec strings into decimal degrees."""
    ra = parse_coordinate(ra_str)
    dec = parse_coordinate(dec_str)
    
    # Convert RA hours to degrees if it looks like hours (< 24)
    if ra <= 24.0 and ':' in ra_str:
        ra *= 15.0
    
    return ra, dec


def format_coordinate(value, is_ra=False, format_type='sexagesimal', ndec=3):
    """Format coordinate for output."""
    if format_type == 'degrees':
        return f"{value:.{ndec}f}"
    
    # Sexagesimal format
    if is_ra:
        # RA in hours
        hours = value / 15.0
        h = int(abs(hours))
        m = int((abs(hours) - h) * 60)
        s = ((abs(hours) - h) * 60 - m) * 60
        return f"{h:02d}:{m:02d}:{s:0{4+ndec}.{ndec}f}"
    else:
        # Dec in degrees
        sign = '-' if value < 0 else '+'
        deg = int(abs(value))
        min_val = int((abs(value) - deg) * 60)
        sec = ((abs(value) - deg) * 60 - min_val) * 60
        return f"{sign}{deg:02d}:{min_val:02d}:{sec:0{4+ndec}.{ndec}f}"


def read_wcs_from_file(filename, verbose=False):
    """Read WCS from FITS file."""
    try:
        with fits.open(filename) as hdulist:
            # Try to find HDU with WCS
            for i, hdu in enumerate(hdulist):
                try:
                    w = wcs.WCS(hdu.header)
                    if w.has_celestial:
                        if verbose:
                            print(f"Found WCS in HDU {i}")
                        return w, hdu.header
                except Exception:
                    continue
            
            # If no celestial WCS found, try primary HDU anyway
            w = wcs.WCS(hdulist[0].header)
            return w, hdulist[0].header
            
    except Exception as e:
        raise RuntimeError(f"Cannot read WCS from {filename}: {e}")


def sky_to_xy(ra, dec, w, coord_sys='icrs'):
    """Convert sky coordinates to pixel coordinates."""
    try:
        if coord_sys.lower() in ['b1950', 'fk4']:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='fk4')
        elif coord_sys.lower() in ['j2000', 'fk5', 'icrs']:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        elif coord_sys.lower() == 'galactic':
            coord = SkyCoord(l=ra*u.deg, b=dec*u.deg, frame='galactic')
        elif coord_sys.lower() == 'ecliptic':
            coord = SkyCoord(lon=ra*u.deg, lat=dec*u.deg, frame='geocentrictrueecliptic')
        else:
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        
        # Convert to ICRS for WCS transformation
        coord_icrs = coord.transform_to('icrs')
        x, y = w.world_to_pixel(coord_icrs)
        
        # Check if position is within image bounds
        offscale = 0
        if hasattr(w, '_naxis1') and hasattr(w, '_naxis2'):
            if x < 0 or x > w._naxis1 or y < 0 or y > w._naxis2:
                offscale = 2  # Off image
        elif x < 0 or y < 0:
            offscale = 1  # Offscale
            
        return float(x), float(y), offscale
        
    except Exception as e:
        if 'world_to_pixel' in str(e):
            return float('nan'), float('nan'), 1
        raise e


def xy_to_sky(x, y, w, output_sys='icrs', format_type='sexagesimal'):
    """Convert pixel coordinates to sky coordinates."""
    try:
        coord = w.pixel_to_world(x, y)
        
        # Convert to requested output system
        if output_sys.lower() in ['b1950', 'fk4']:
            coord_out = coord.transform_to('fk4')
            ra, dec = coord_out.ra.deg, coord_out.dec.deg
            sys_name = 'B1950'
        elif output_sys.lower() in ['j2000', 'fk5', 'icrs']:
            coord_out = coord.transform_to('icrs')
            ra, dec = coord_out.ra.deg, coord_out.dec.deg
            sys_name = 'J2000'
        elif output_sys.lower() == 'galactic':
            coord_out = coord.transform_to('galactic')
            ra, dec = coord_out.l.deg, coord_out.b.deg
            sys_name = 'galactic'
        elif output_sys.lower() == 'ecliptic':
            coord_out = coord.transform_to('geocentrictrueecliptic')
            ra, dec = coord_out.lon.deg, coord_out.lat.deg
            sys_name = 'ecliptic'
        else:
            coord_out = coord.transform_to('icrs')
            ra, dec = coord_out.ra.deg, coord_out.dec.deg
            sys_name = 'J2000'
            
        return ra, dec, sys_name
        
    except Exception as e:
        if 'pixel_to_world' in str(e):
            return float('nan'), float('nan'), 'unknown'
        raise e


def process_sky2xy(args):
    """Process sky2xy conversion."""
    # Read WCS
    w, header = read_wcs_from_file(args.fits_file, args.verbose)
    
    # Determine coordinate system
    coord_sys = 'icrs'  # default
    if args.b1950:
        coord_sys = 'b1950'
    elif args.j2000:
        coord_sys = 'j2000'
    elif args.galactic:
        coord_sys = 'galactic'
    elif args.ecliptic:
        coord_sys = 'ecliptic'
    
    # Process coordinates
    results = []
    
    if args.coords:
        # Command line coordinates
        coords = args.coords
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                try:
                    ra, dec = parse_ra_dec(coords[i], coords[i+1])
                    x, y, offscale = sky_to_xy(ra, dec, w, coord_sys)
                    
                    if args.output_only == 'x':
                        print(f"{x:.{args.ndec}f}")
                    elif args.output_only == 'y':
                        print(f"{y:.{args.ndec}f}")
                    else:
                        status = ""
                        if offscale == 2:
                            status = " (off image)"
                        elif offscale == 1:
                            status = " (offscale)"
                        
                        if args.verbose:
                            print(f"{coords[i]} {coords[i+1]} {coord_sys} -> {ra:.5f} {dec:.5f} -> {x:.{args.ndec}f} {y:.{args.ndec}f}{status}")
                        else:
                            print(f"{coords[i]} {coords[i+1]} {coord_sys} -> {x:.{args.ndec}f} {y:.{args.ndec}f}{status}")
                            
                except Exception as e:
                    print(f"Error processing {coords[i]} {coords[i+1]}: {e}")
    
    elif args.list_file:
        # File input
        try:
            with open(args.list_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            ra, dec = parse_ra_dec(parts[0], parts[1])
                            coord_sys_line = coord_sys
                            if len(parts) > 2:
                                coord_sys_line = parts[2].lower()
                            
                            x, y, offscale = sky_to_xy(ra, dec, w, coord_sys_line)
                            
                            status = ""
                            if offscale == 2:
                                status = " (off image)"
                            elif offscale == 1:
                                status = " (offscale)"
                            
                            if args.verbose:
                                print(f"{parts[0]} {parts[1]} {coord_sys_line} -> {ra:.5f} {dec:.5f} -> {x:.{args.ndec}f} {y:.{args.ndec}f}{status}")
                            else:
                                print(f"{parts[0]} {parts[1]} {coord_sys_line} -> {x:.{args.ndec}f} {y:.{args.ndec}f}{status}")
                                
                        except Exception as e:
                            print(f"Error processing line {line_num}: {e}")
                            
        except FileNotFoundError:
            print(f"Error: Cannot read file {args.list_file}")
            return 1
    
    else:
        # Read from stdin
        if args.verbose:
            print("Reading coordinates from stdin (ra dec [sys])...")
        
        for line_num, line in enumerate(sys.stdin, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    ra, dec = parse_ra_dec(parts[0], parts[1])
                    coord_sys_line = coord_sys
                    if len(parts) > 2:
                        coord_sys_line = parts[2].lower()
                    
                    x, y, offscale = sky_to_xy(ra, dec, w, coord_sys_line)
                    
                    if args.output_only == 'x':
                        print(f"{x:.{args.ndec}f}")
                    elif args.output_only == 'y':
                        print(f"{y:.{args.ndec}f}")
                    else:
                        status = ""
                        if offscale == 2:
                            status = " (off image)"
                        elif offscale == 1:
                            status = " (offscale)"
                        
                        print(f"{x:.{args.ndec}f} {y:.{args.ndec}f}{status}")
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
    
    return 0


def process_xy2sky(args):
    """Process xy2sky conversion."""
    # Read WCS
    w, header = read_wcs_from_file(args.fits_file, args.verbose)
    
    # Determine output coordinate system
    output_sys = 'icrs'  # default
    if args.b1950:
        output_sys = 'b1950'
    elif args.j2000:
        output_sys = 'j2000'
    elif args.galactic:
        output_sys = 'galactic'
    elif args.ecliptic:
        output_sys = 'ecliptic'
    
    # Determine output format
    format_type = 'degrees' if args.degrees else 'sexagesimal'
    
    # Print header if requested
    if args.print_header and not args.output_only:
        if format_type == 'degrees':
            if output_sys == 'galactic':
                print("# Longitude  Latitude   Sys       X        Y")
            elif output_sys == 'ecliptic':
                print("# Longitude  Latitude   Sys       X        Y")
            else:
                print("#     RA         Dec     Sys       X        Y")
        else:
            if output_sys == 'galactic':
                print("# Longitude     Latitude    Sys       X        Y")
            elif output_sys == 'ecliptic':
                print("# Longitude     Latitude    Sys       X        Y")
            else:
                print("#     RA           Dec      Sys       X        Y")
    
    # Process coordinates
    if args.coords:
        # Command line coordinates
        coords = args.coords
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                try:
                    x = float(coords[i])
                    y = float(coords[i+1])
                    ra, dec, sys_name = xy_to_sky(x, y, w, output_sys, format_type)
                    
                    if args.output_only == 'r':
                        if format_type == 'degrees':
                            print(f"{ra:.{args.ndec}f}")
                        else:
                            print(format_coordinate(ra, is_ra=True, format_type=format_type, ndec=args.ndec))
                    elif args.output_only == 'd':
                        if format_type == 'degrees':
                            print(f"{dec:.{args.ndec}f}")
                        else:
                            print(format_coordinate(dec, is_ra=False, format_type=format_type, ndec=args.ndec))
                    elif args.output_only == 's':
                        print(sys_name)
                    else:
                        if format_type == 'degrees':
                            ra_str = f"{ra:.{args.ndec}f}"
                            dec_str = f"{dec:.{args.ndec}f}"
                        else:
                            ra_str = format_coordinate(ra, is_ra=True, format_type=format_type, ndec=args.ndec)
                            dec_str = format_coordinate(dec, is_ra=False, format_type=format_type, ndec=args.ndec)
                        
                        if args.append_input:
                            print(f"{ra_str} {dec_str} {sys_name} {x:.3f} {y:.3f}")
                        else:
                            if args.verbose:
                                print(f"{ra_str} {dec_str} {sys_name} <- {x:.3f} {y:.3f}")
                            else:
                                print(f"{ra_str} {dec_str} {sys_name}")
                            
                except Exception as e:
                    print(f"Error processing {coords[i]} {coords[i+1]}: {e}")
    
    elif args.list_file:
        # File input
        try:
            with open(args.list_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            ra, dec, sys_name = xy_to_sky(x, y, w, output_sys, format_type)
                            
                            if format_type == 'degrees':
                                ra_str = f"{ra:.{args.ndec}f}"
                                dec_str = f"{dec:.{args.ndec}f}"
                            else:
                                ra_str = format_coordinate(ra, is_ra=True, format_type=format_type, ndec=args.ndec)
                                dec_str = format_coordinate(dec, is_ra=False, format_type=format_type, ndec=args.ndec)
                            
                            if args.append_input:
                                print(f"{ra_str} {dec_str} {sys_name} {line}")
                            else:
                                print(f"{ra_str} {dec_str} {sys_name}")
                                
                        except Exception as e:
                            print(f"Error processing line {line_num}: {e}")
                            
        except FileNotFoundError:
            print(f"Error: Cannot read file {args.list_file}")
            return 1
    
    else:
        # Read from stdin
        if args.verbose:
            print("Reading pixel coordinates from stdin (x y)...")
        
        for line_num, line in enumerate(sys.stdin, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    ra, dec, sys_name = xy_to_sky(x, y, w, output_sys, format_type)
                    
                    if args.output_only == 'r':
                        if format_type == 'degrees':
                            print(f"{ra:.{args.ndec}f}")
                        else:
                            print(format_coordinate(ra, is_ra=True, format_type=format_type, ndec=args.ndec))
                    elif args.output_only == 'd':
                        if format_type == 'degrees':
                            print(f"{dec:.{args.ndec}f}")
                        else:
                            print(format_coordinate(dec, is_ra=False, format_type=format_type, ndec=args.ndec))
                    elif args.output_only == 's':
                        print(sys_name)
                    else:
                        if format_type == 'degrees':
                            ra_str = f"{ra:.{args.ndec}f}"
                            dec_str = f"{dec:.{args.ndec}f}"
                        else:
                            ra_str = format_coordinate(ra, is_ra=True, format_type=format_type, ndec=args.ndec)
                            dec_str = format_coordinate(dec, is_ra=False, format_type=format_type, ndec=args.ndec)
                        
                        print(f"{ra_str} {dec_str} {sys_name}")
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
    
    return 0


def main():
    # Detect mode from program name
    auto_mode = detect_mode()
    
    parser = argparse.ArgumentParser(
        description="Convert between pixel and world coordinates using FITS WCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sky2xy image.fits "12:34:56.7" "+12:34:56"     # Convert RA/Dec to X/Y
  xy2sky image.fits 100.5 200.3                  # Convert X/Y to RA/Dec
  echo "100 200" | xy2sky image.fits             # Pipe coordinates
  sky2xy -j image.fits 185.5 12.5                # Input as J2000 degrees
  xy2sky -d image.fits 100 200                   # Output in degrees
  
For filter mode (no coordinates given), reads from stdin:
  cat coords.txt | sky2xy image.fits             # RA Dec [sys] per line
  cat pixels.txt | xy2sky image.fits             # X Y per line
        """
    )
    
    # Mode selection
    parser.add_argument('-m', '--mode', choices=['sky2xy', 'xy2sky'],
                       help='Conversion mode (auto-detected from program name)')
    
    # Common arguments
    parser.add_argument('fits_file', nargs='?', help='FITS file with WCS')
    parser.add_argument('coords', nargs='*', help='Coordinates to convert')
    
    parser.add_argument('-f', '--file', dest='list_file',
                       help='Read coordinates from file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('-n', '--ndec', type=int, default=3,
                       help='Number of decimal places in output')
    
    # Coordinate system options
    parser.add_argument('-b', '--b1950', action='store_true',
                       help='Use B1950 (FK4) coordinates')
    parser.add_argument('-j', '--j2000', action='store_true',
                       help='Use J2000 (FK5) coordinates')
    parser.add_argument('-g', '--galactic', action='store_true',
                       help='Use galactic coordinates')
    parser.add_argument('-e', '--ecliptic', action='store_true',
                       help='Use ecliptic coordinates')
    
    # xy2sky specific options
    parser.add_argument('-d', '--degrees', action='store_true',
                       help='Output coordinates in degrees (xy2sky)')
    parser.add_argument('-a', '--append-input', action='store_true',
                       help='Append input line to output (xy2sky)')
    parser.add_argument('-p', '--print-header', action='store_true',
                       help='Print column headers (xy2sky)')
    parser.add_argument('-o', '--output-only', choices=['r', 'd', 's', 'x', 'y'],
                       help='Output only RA/Dec/Sys (xy2sky) or X/Y (sky2xy)')
    
    parser.add_argument('--version', action='version',
                       version='wcsconv.py 1.0 - Unified WCS coordinate converter')
    
    args = parser.parse_args()
    
    # Handle case where no arguments provided
    if not args.fits_file:
        parser.print_help()
        return 1
    
    # Determine mode
    if args.mode:
        mode = args.mode
    elif auto_mode:
        mode = auto_mode
    else:
        print("Error: Cannot determine conversion mode. Use -m sky2xy or -m xy2sky")
        return 1
    
    # Check if FITS file exists
    if not os.path.exists(args.fits_file):
        print(f"Error: FITS file {args.fits_file} not found")
        return 1
    
    # Process based on mode
    try:
        if mode == 'sky2xy':
            return process_sky2xy(args)
        else:
            return process_xy2sky(args)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
