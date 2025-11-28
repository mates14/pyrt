#!/usr/bin/env python3
"""
edhead.py - Edit FITS file headers
Python replacement for WCSTools edhead with proper multi-HDU support

Author: Python rewrite of Doug Mink's original C code
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import astropy.io.fits as fits
except ImportError:
    print("Error: astropy is required. Install with: pip install astropy")
    sys.exit(1)


def get_default_editor():
    """Get default editor from environment or system."""
    editor = os.environ.get('EDITOR')
    if editor:
        return editor
    
    # Try common editors
    for cmd in ['vim', 'vi', 'nano', 'emacs']:
        if subprocess.run(['which', cmd], capture_output=True).returncode == 0:
            return cmd
    
    raise RuntimeError("No editor found. Set EDITOR environment variable.")


def list_hdus(filename, verbose=False):
    """List all HDUs in a FITS file."""
    try:
        with fits.open(filename) as hdulist:
            print(f"\nFITS file: {filename}")
            print(f"Number of HDUs: {len(hdulist)}")
            print("-" * 50)
            
            for i, hdu in enumerate(hdulist):
                hdu_type = type(hdu).__name__
                
                if hasattr(hdu, 'data') and hdu.data is not None:
                    if hasattr(hdu.data, 'shape'):
                        shape = f", shape: {hdu.data.shape}"
                    else:
                        shape = ", data: present"
                else:
                    shape = ", no data"
                
                # Get extension name if present
                extname = hdu.header.get('EXTNAME', '')
                if extname:
                    extname = f" ({extname})"
                
                print(f"HDU {i}: {hdu_type}{extname}{shape}")
                
                if verbose and i == 0:
                    # Show primary header info
                    naxis = hdu.header.get('NAXIS', 0)
                    if naxis > 0:
                        dims = [hdu.header.get(f'NAXIS{j}', 0) for j in range(1, naxis+1)]
                        print(f"       Dimensions: {dims}")
                    bitpix = hdu.header.get('BITPIX', 0)
                    if bitpix:
                        print(f"       BITPIX: {bitpix}")
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return False
    
    return True


def format_fits_value(value):
    """Format a value for FITS header text representation."""
    if isinstance(value, bool):
        return 'T' if value else 'F'
    elif isinstance(value, str):
        return f"'{value}'"
    else:
        return str(value)


def parse_fits_value(value_str):
    """Parse a FITS value from text representation."""
    value_str = value_str.strip()
    
    # Handle boolean values
    if value_str == 'T':
        return True
    elif value_str == 'F':
        return False
    
    # Handle quoted strings
    if value_str.startswith("'") and value_str.endswith("'"):
        return value_str[1:-1]  # Remove quotes
    
    # Handle numbers
    try:
        if '.' in value_str or 'e' in value_str.lower() or 'E' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        # If all else fails, return as string
        return value_str


def headers_equal(header1, header2):
    """Compare two FITS headers for semantic equality."""
    # Quick check: if different number of cards, they're different
    if len(header1) != len(header2):
        return False
    
    # Compare each card
    for i, (card1, card2) in enumerate(zip(header1.cards, header2.cards)):
        # Compare keyword, value, and comment
        if (card1.keyword != card2.keyword or 
            card1.value != card2.value or 
            card1.comment != card2.comment):
            return False
    
    return True


def write_header_to_file(header, filepath):
    """Write FITS header to text file for editing."""
    with open(filepath, 'w') as f:
        for card in header.cards:
            # Write each card as it appears, handling comments properly
            if card.keyword == 'COMMENT' or card.keyword == 'HISTORY':
                f.write(f"{card.keyword} {card.value}\n")
            else:
                # Format: KEYWORD= value with = at column 9, comments at column 40
                keyword_equals = f"{card.keyword:<8}="
                value_str = format_fits_value(card.value)
                
                if card.comment:
                    # Calculate spacing to align comment at column 40
                    value_part = f"{keyword_equals} {value_str}"
                    if len(value_part) < 39:
                        spaces = " " * (39 - len(value_part))
                        f.write(f"{value_part}{spaces}/ {card.comment}\n")
                    else:
                        # If value is too long, put comment right after with single space
                        f.write(f"{value_part} / {card.comment}\n")
                else:
                    f.write(f"{keyword_equals} {value_str}\n")


def read_header_from_file(filepath):
    """Read edited header from text file and convert back to FITS header."""
    new_header = fits.Header()
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip('\n\r')
            if not line.strip():
                continue
                
            try:
                # Handle COMMENT and HISTORY cards
                if line.startswith('COMMENT '):
                    new_header.add_comment(line[8:])
                elif line.startswith('HISTORY '):
                    new_header.add_history(line[8:])
                else:
                    # Parse regular cards
                    if '=' in line:
                        parts = line.split('=', 1)
                        keyword = parts[0].strip()
                        value_comment = parts[1].strip()
                        
                        # Split value and comment
                        if '/' in value_comment:
                            value_part, comment = value_comment.split('/', 1)
                            comment = comment.strip()
                        else:
                            value_part = value_comment
                            comment = ''
                        
                        # Parse value
                        value = parse_fits_value(value_part)
                        
                        new_header[keyword] = (value, comment)
                    else:
                        # Line without '=' - treat as comment
                        new_header.add_comment(line)
                        
            except Exception as e:
                print(f"Warning: Could not parse line {line_num}: {line}")
                print(f"Error: {e}")
                continue
    
    return new_header


def edit_header(filename, hdu_num=None, editor=None, new_file=False, verbose=False):
    """Edit FITS header using external editor."""
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return False
    
    try:
        # Open file and check HDU structure
        with fits.open(filename) as hdulist:
            num_hdus = len(hdulist)
            
            # Handle multi-HDU files
            if num_hdus > 1 and hdu_num is None:
                print(f"Multi-HDU file detected ({num_hdus} HDUs).")
                print("Use -l to list HDUs, or -u N to edit specific HDU.")
                list_hdus(filename, verbose)
                return True
            
            # Validate HDU number
            if hdu_num is not None:
                if hdu_num >= num_hdus or hdu_num < 0:
                    print(f"Error: HDU {hdu_num} does not exist. File has {num_hdus} HDUs (0-{num_hdus-1})")
                    return False
                target_hdu = hdu_num
            else:
                target_hdu = 0
            
            # Get original header
            original_header = hdulist[target_hdu].header.copy()
            
        # Create temporary file for editing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name
            write_header_to_file(original_header, temp_path)
        
        if verbose:
            print(f"Editing HDU {target_hdu} of {filename}")
            print(f"Temporary file: {temp_path}")
            
        try:
            # Launch editor
            editor_cmd = editor or get_default_editor()
            if verbose:
                print(f"Editor command: {editor_cmd}")
            
            result = subprocess.run([editor_cmd, temp_path])
            if result.returncode != 0:
                print(f"Editor exited with code {result.returncode}")
                return False
            
            # Read edited header
            new_header = read_header_from_file(temp_path)
            
            # Check if header actually changed
            if headers_equal(original_header, new_header):
                if verbose:
                    print("No changes detected - file left unchanged")
                return True
            
            # Determine output filename
            if new_file or (len(new_header) * 80 > len(original_header) * 80 and 
                           original_header.get('NAXIS', 0) == 0):
                # Create new file
                path = Path(filename)
                if path.suffix.lower() in ['.fits', '.fit', '.fts']:
                    output_file = path.with_name(f"{path.stem}e{path.suffix}")
                else:
                    output_file = Path(f"{filename}e")
                
                if verbose:
                    print(f"Creating new file: {output_file}")
            else:
                output_file = filename
                if verbose:
                    print(f"Updating existing file: {output_file}")
            
            # Write the modified file
            if str(output_file) != filename:
                # Copy entire file and modify specific HDU
                with fits.open(filename) as hdulist:
                    hdulist[target_hdu].header = new_header
                    hdulist.writeto(output_file, overwrite=True)
            else:
                # Update in place
                with fits.open(filename, mode='update') as hdulist:
                    hdulist[target_hdu].header = new_header
                    hdulist.flush()
            
            if verbose:
                print(f"Successfully updated {output_file}")
            
            return True
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
                
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Edit FITS file headers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.fits              # Edit primary HDU of single-HDU file
  %(prog)s -l image.fits           # List all HDUs in file
  %(prog)s -u 1 image.fits         # Edit HDU 1 (first extension)
  %(prog)s -n -e nano image.fits   # Create new file using nano editor
  %(prog)s -v image.fits           # Verbose output
        """
    )
    
    parser.add_argument('files', nargs='+', help='FITS files to edit')
    parser.add_argument('-l', '--list', action='store_true',
                       help='List HDUs and exit')
    parser.add_argument('-u', '--hdu', type=int, metavar='N',
                       help='Edit HDU number N (0=primary)')
    parser.add_argument('-e', '--editor', metavar='EDITOR',
                       help='Specify editor (overrides EDITOR env var)')
    parser.add_argument('-n', '--new', action='store_true',
                       help='Write to new file instead of overwriting')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--version', action='version', 
                       version='edhead.py 1.0 - Python FITS header editor')
    
    args = parser.parse_args()
    
    success = True
    
    for filename in args.files:
        if args.list:
            if not list_hdus(filename, args.verbose):
                success = False
        else:
            if not edit_header(filename, args.hdu, args.editor, 
                             args.new, args.verbose):
                success = False
        
        if args.verbose and len(args.files) > 1:
            print()  # Blank line between files
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
