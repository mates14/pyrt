#!/usr/bin/env python3
"""
cphead.py - Copy FITS header keywords between files
Python replacement for WCSTools cphead with proper multi-HDU support

Author: Python rewrite of Jessica Mink's original C code
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import astropy.io.fits as fits
except ImportError:
    print("Error: astropy is required. Install with: pip install astropy")
    sys.exit(1)


def get_wcs_keywords():
    """Return comprehensive list of WCS keywords including SIP distortion."""
    # Basic WCS keywords
    wcs_keywords = [
        # Core WCS
        'WCSAXES', 'WCSNAME', 'WCSNAMEA', 'WCSNAMEB', 'WCSNAMEN',
        'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
        'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
        'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2',
        'CNAME1', 'CNAME2',
        
        # CD matrix (modern)
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
        
        # PC matrix + CDELT (alternative)
        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
        
        # Legacy/compatibility
        'PC001001', 'PC001002', 'PC002001', 'PC002002',
        
        # Celestial coordinate system
        'EQUINOX', 'EPOCH', 'RADESYS', 'RADECSYS',
        'LATPOLE', 'LONPOLE',
        
        # Common non-standard but useful
        'RA', 'DEC', 'SECPIX', 'SECPIX1', 'SECPIX2', 'IMWCS',
        
        # SIP distortion keywords
        'A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER',
        'A_DMAX', 'B_DMAX', 'AP_DMAX', 'BP_DMAX',
        
        # TPV projection distortion
        'PV1_0', 'PV1_1', 'PV1_2', 'PV1_4', 'PV1_5', 'PV1_6',
        'PV2_0', 'PV2_1', 'PV2_2', 'PV2_4', 'PV2_5', 'PV2_6',
    ]
    
    # Add SIP coefficients A_i_j, B_i_j, AP_i_j, BP_i_j (up to 9th order)
    for order in range(10):
        for i in range(order + 1):
            j = order - i
            wcs_keywords.extend([
                f'A_{i}_{j}', f'B_{i}_{j}',
                f'AP_{i}_{j}', f'BP_{i}_{j}'
            ])
    
    # Add more PV parameters (up to PV1_39, PV2_39 as per FITS standard)
    for axis in [1, 2]:
        for param in range(40):
            wcs_keywords.append(f'PV{axis}_{param}')
    
    # Add PS parameters (less common)
    for axis in [1, 2]:
        for param in range(10):
            wcs_keywords.append(f'PS{axis}_{param}')
    
    # Add alternate WCS versions (A, B, C, etc.)
    for alt in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        wcs_keywords.extend([
            f'CTYPE1{alt}', f'CTYPE2{alt}', f'CUNIT1{alt}', f'CUNIT2{alt}',
            f'CRVAL1{alt}', f'CRVAL2{alt}', f'CRPIX1{alt}', f'CRPIX2{alt}',
            f'CDELT1{alt}', f'CDELT2{alt}', f'CROTA1{alt}', f'CROTA2{alt}',
            f'CD1_1{alt}', f'CD1_2{alt}', f'CD2_1{alt}', f'CD2_2{alt}',
            f'PC1_1{alt}', f'PC1_2{alt}', f'PC2_1{alt}', f'PC2_2{alt}',
            f'WCSNAME{alt}', f'CNAME1{alt}', f'CNAME2{alt}'
        ])
    
    return wcs_keywords


def get_all_wcs_keywords():
    """Return all possible WCS-related keywords for removal."""
    return get_wcs_keywords() + [
        # Additional legacy and variant keywords
        'OBJECT', 'OBJCTRA', 'OBJCTDEC', 'OBJCTX', 'OBJCTY',
        'XPIXELSZ', 'YPIXELSZ', 'PIXSCALE', 'PIXSCAL1', 'PIXSCAL2',
        'ORIENTAT', 'POSANG', 'INSTRUME', 'DETECTOR',
        'TELRA', 'TELDEC', 'TELALT', 'TELAZ',
        'PLATEID', 'EXPOSURE', 'EXPTIME',
        'AIRMASS', 'SECZ', 'ZD',
        'OBSRA', 'OBSDEC', 'OBSEPOCH',
        'XREFVAL', 'YREFVAL', 'XREFPIX', 'YREFPIX',
        'XINC', 'YINC', 'XROT', 'YROT'
    ]


def remove_wcs_keywords(header, verbose=False):
    """Remove all WCS-related keywords from header."""
    wcs_keywords = get_all_wcs_keywords()
    removed_count = 0
    
    # Make a copy of keywords to avoid modifying during iteration
    keywords_to_check = list(header.keys())
    
    for keyword in keywords_to_check:
        # Check exact matches and pattern matches
        should_remove = False
        
        if keyword in wcs_keywords:
            should_remove = True
        elif keyword.startswith(('CTYPE', 'CRVAL', 'CRPIX', 'CDELT', 'CROTA',
                                'CD1_', 'CD2_', 'PC1_', 'PC2_', 'PV1_', 'PV2_',
                                'PS1_', 'PS2_', 'A_', 'B_', 'AP_', 'BP_',
                                'CUNIT', 'CNAME', 'WCSNAME')):
            should_remove = True
        
        if should_remove:
            if verbose:
                print(f"Removing WCS keyword: {keyword}")
            del header[keyword]
            removed_count += 1
    
    return removed_count


def read_keyword_list(filename):
    """Read keywords from a file, one per line."""
    keywords = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                keyword = line.strip()
                if keyword and not keyword.startswith('#'):
                    keywords.append(keyword.upper())
    except FileNotFoundError:
        print(f"Error: Keyword file {filename} not found")
        sys.exit(1)
    return keywords


def read_file_list(filename):
    """Read filenames from a file, one per line."""
    files = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                filename = line.strip()
                if filename and not filename.startswith('#'):
                    files.append(filename)
    except FileNotFoundError:
        print(f"Error: File list {filename} not found")
        sys.exit(1)
    return files


def list_hdus(filename):
    """List HDUs in a FITS file."""
    try:
        with fits.open(filename) as hdulist:
            print(f"\nFITS file: {filename}")
            print(f"Number of HDUs: {len(hdulist)}")
            print("-" * 40)
            for i, hdu in enumerate(hdulist):
                hdu_type = type(hdu).__name__
                extname = hdu.header.get('EXTNAME', '')
                if extname:
                    print(f"HDU {i}: {hdu_type} ({extname})")
                else:
                    print(f"HDU {i}: {hdu_type}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")


def copy_header_keywords(source_file, dest_file, keywords, source_hdu=0, dest_hdu=0,
                        copy_all=False, new_file=False, add_history=False, 
                        precision=None, remove_wcs=False, verbose=False):
    """Copy specified keywords from source to destination file."""
    
    # Read source header (only if not removing WCS)
    if not remove_wcs:
        try:
            with fits.open(source_file) as src_hdulist:
                if source_hdu >= len(src_hdulist):
                    print(f"Error: Source HDU {source_hdu} does not exist in {source_file}")
                    return False
                source_header = src_hdulist[source_hdu].header
        except Exception as e:
            print(f"Error reading source file {source_file}: {e}")
            return False
    
    # Determine output filename
    if new_file:
        path = Path(dest_file)
        if path.suffix.lower() in ['.fits', '.fit', '.fts']:
            output_file = path.with_name(f"{path.stem}e{path.suffix}")
        else:
            output_file = Path(f"{dest_file}e")
    else:
        output_file = dest_file
    
    # Read destination file and update
    try:
        with fits.open(dest_file) as dest_hdulist:
            if dest_hdu >= len(dest_hdulist):
                print(f"Error: Destination HDU {dest_hdu} does not exist in {dest_file}")
                return False
            
            dest_header = dest_hdulist[dest_hdu].header
            
            if remove_wcs:
                # Remove WCS keywords from destination
                removed_count = remove_wcs_keywords(dest_header, verbose)
                if verbose:
                    print(f"Removed {removed_count} WCS keywords")
                
                # Add history if requested
                if add_history:
                    timestamp = time.strftime('%Y-%m-%d %H:%M')
                    history_msg = f"cphead.py {timestamp}: Removed WCS keywords ({removed_count} total)"
                    dest_header.add_history(history_msg)
                
                copied_count = 0  # For consistency in return message
            
            elif copy_all:
                # Replace entire header (preserve required keywords)
                essential_keywords = ['SIMPLE', 'BITPIX', 'NAXIS']
                essential_keywords.extend([f'NAXIS{i}' for i in range(1, dest_header.get('NAXIS', 0) + 1)])
                essential_keywords.extend(['EXTEND', 'PCOUNT', 'GCOUNT', 'TFIELDS'])
                
                # Save essential keywords from destination
                essential_cards = {}
                for key in essential_keywords:
                    if key in dest_header:
                        essential_cards[key] = dest_header[key]
                
                # Copy source header
                new_header = source_header.copy()
                
                # Restore essential keywords
                for key, value in essential_cards.items():
                    if key in ['SIMPLE', 'BITPIX', 'NAXIS'] + [f'NAXIS{i}' for i in range(1, 10)]:
                        new_header[key] = value
                
                dest_hdulist[dest_hdu].header = new_header
                copied_count = len(source_header)
                
                if verbose:
                    print(f"Copied entire header ({copied_count} keywords)")
                
                # Add history if requested
                if add_history:
                    timestamp = time.strftime('%Y-%m-%d %H:%M')
                    history_msg = f"cphead.py {timestamp}: Copied entire header from {source_file}"
                    dest_hdulist[dest_hdu].header.add_history(history_msg)
                
            else:
                # Copy specific keywords
                copied_count = 0
                missing_keywords = []
                
                for keyword in keywords:
                    if keyword in source_header:
                        value = source_header[keyword]
                        comment = source_header.comments[keyword]
                        
                        # Handle precision for numeric values
                        if precision is not None and isinstance(value, (int, float)):
                            if isinstance(value, float):
                                # Format with specified precision
                                formatted_value = round(value, precision)
                                dest_header[keyword] = (formatted_value, comment)
                            else:
                                dest_header[keyword] = (value, comment)
                        else:
                            dest_header[keyword] = (value, comment)
                        
                        copied_count += 1
                        
                        if verbose:
                            print(f"Copied {keyword} = {value}")
                    else:
                        missing_keywords.append(keyword)
                        if verbose:
                            print(f"Keyword {keyword} not found in source")
                
                if verbose and missing_keywords:
                    print(f"Missing keywords: {', '.join(missing_keywords)}")
                
                # Add history if requested
                if add_history:
                    timestamp = time.strftime('%Y-%m-%d %H:%M')
                    if len(keywords) <= 5:
                        kw_list = ', '.join(keywords[:5])
                    else:
                        kw_list = ', '.join(keywords[:5]) + f' and {len(keywords)-5} more'
                    history_msg = f"cphead.py {timestamp}: Copied {kw_list} from {source_file}"
                    dest_header.add_history(history_msg)
            
            # Write output file
            if str(output_file) != dest_file:
                dest_hdulist.writeto(output_file, overwrite=True)
                if verbose:
                    print(f"Created new file: {output_file}")
            else:
                dest_hdulist.writeto(output_file, overwrite=True)
                if verbose:
                    print(f"Updated file: {output_file}")
            
            if verbose and not remove_wcs:
                print(f"Successfully copied {copied_count} keywords")
            
            return True
            
    except Exception as e:
        print(f"Error processing destination file {dest_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Copy FITS header keywords from source to destination files, or remove WCS keywords",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s source.fits dest.fits OBJECT EXPTIME    # Copy specific keywords
  %(prog)s -w source.fits dest.fits                # Copy comprehensive WCS
  %(prog)s -r dest.fits                            # Remove all WCS keywords
  %(prog)s -r @filelist.txt                        # Remove WCS from file list
  %(prog)s -a source.fits dest.fits                # Copy entire header
  %(prog)s source.fits @filelist.txt FILTER       # Copy to multiple files
  %(prog)s source.fits dest.fits @keywords.txt    # Copy keywords from file
  %(prog)s -u 1 -t 0 source.fits dest.fits CD1_1  # Copy from HDU 1 to HDU 0
        """
    )
    
    parser.add_argument('source', nargs='?', help='Source FITS file (not needed for -r)')
    parser.add_argument('destinations', nargs='*', 
                       help='Destination files and/or keywords')
    
    parser.add_argument('-a', '--all', action='store_true',
                       help='Copy entire header (overwrite destination header)')
    parser.add_argument('-w', '--wcs', action='store_true',
                       help='Copy comprehensive WCS keywords (including SIP distortion)')
    parser.add_argument('-r', '--remove-wcs', action='store_true',
                       help='Remove all WCS keywords from destination files')
    parser.add_argument('-n', '--new', action='store_true',
                       help='Create new files instead of overwriting')
    parser.add_argument('-H', '--history', action='store_true',
                       help='Add HISTORY record of operation')
    parser.add_argument('-p', '--precision', type=int, metavar='N',
                       help='Number of decimal places for numeric values')
    parser.add_argument('-u', '--source-hdu', type=int, default=0, metavar='N',
                       help='Source HDU number (default: 0)')
    parser.add_argument('-t', '--dest-hdu', type=int, default=0, metavar='N',
                       help='Destination HDU number (default: 0)')
    parser.add_argument('-l', '--list', action='store_true',
                       help='List HDUs in source file and exit')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--version', action='version',
                       version='cphead.py 1.0 - Python FITS keyword copier')
    
    args = parser.parse_args()
    
    # Handle case where no arguments provided
    if not args.source and not args.destinations and not args.remove_wcs and not args.list:
        parser.print_help()
        return 1
    
    # Handle list option
    if args.list:
        if not args.source:
            print("Error: Source file required for -l option")
            return 1
        list_hdus(args.source)
        return 0
    
    # Special case: remove WCS doesn't need source file
    if args.remove_wcs:
        # For WCS removal, all arguments are destination files
        all_args = []
        if args.source:
            all_args.append(args.source)
        all_args.extend(args.destinations)
        
        if not all_args:
            print("Error: No destination files specified for WCS removal")
            return 1
        
        # Parse file lists
        files = []
        for item in all_args:
            if item.startswith('@'):
                listfile = item[1:]
                if os.path.exists(listfile):
                    files.extend(read_file_list(listfile))
            else:
                files.append(item)
        
        if not files:
            print("Error: No destination files specified for WCS removal")
            return 1
        
        success_count = 0
        for dest_file in files:
            if args.verbose:
                print(f"\nRemoving WCS keywords from {dest_file}...")
            
            if not os.path.exists(dest_file):
                print(f"Warning: File {dest_file} not found, skipping")
                continue
            
            success = copy_header_keywords(
                None, dest_file, [],  # No source file or keywords needed
                dest_hdu=args.dest_hdu,
                new_file=args.new,
                add_history=args.history,
                remove_wcs=True,
                verbose=args.verbose
            )
            
            if success:
                success_count += 1
        
        if args.verbose:
            print(f"\nSuccessfully processed {success_count}/{len(files)} files")
        return 0 if success_count == len(files) else 1
    
    # For normal operations, source file is required
    if not args.source:
        print("Error: Source file required (or use -r for WCS removal)")
        return 1
    
    # Check if source file exists (for normal copy operations)
    if not os.path.exists(args.source):
        print(f"Error: Source file {args.source} not found")
        return 1
    
    # Parse destinations and keywords
    files = []
    keywords = []
    
    for item in args.destinations:
        if item.startswith('@'):
            # File list or keyword list
            listfile = item[1:]
            if not os.path.exists(listfile):
                print(f"Error: List file {listfile} not found")
                return 1
            
            # Try to determine if it's a file list or keyword list
            # Simple heuristic: if first line looks like a FITS file, it's a file list
            with open(listfile, 'r') as f:
                first_line = f.readline().strip()
                if first_line.lower().endswith(('.fits', '.fit', '.fts')):
                    files.extend(read_file_list(listfile))
                else:
                    keywords.extend(read_keyword_list(listfile))
        elif item.lower().endswith(('.fits', '.fit', '.fts')):
            files.append(item)
        else:
            keywords.append(item.upper())
    
    # Add WCS keywords if requested
    if args.wcs:
        keywords.extend(get_wcs_keywords())
    
    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))
    
    # Check that we have destinations
    if not files:
        print("Error: No destination files specified")
        return 1
    
    # Check that we have keywords or copy_all flag (not needed for remove_wcs)
    if not keywords and not args.all and not args.remove_wcs:
        print("Error: No keywords specified (use -a to copy entire header or -w for WCS)")
        return 1
    
    if args.verbose:
        if not args.remove_wcs and args.source:
            print(f"Source file: {args.source} (HDU {args.source_hdu})")
        print(f"Destination files: {len(files)}")
        if args.remove_wcs:
            print("Operation: Remove all WCS keywords")
        elif args.all:
            print("Operation: Copy entire header")
        else:
            print(f"Keywords to copy: {len(keywords)}")
            if args.verbose and keywords:
                print(f"Keywords: {', '.join(keywords)}")
    
    # Process each destination file
    success_count = 0
    for dest_file in files:
        if args.verbose:
            print(f"\nProcessing {dest_file}...")
        
        if not os.path.exists(dest_file):
            print(f"Warning: Destination file {dest_file} not found, skipping")
            continue
        
        success = copy_header_keywords(
            args.source, dest_file, keywords,
            source_hdu=args.source_hdu,
            dest_hdu=args.dest_hdu,
            copy_all=args.all,
            new_file=args.new,
            add_history=args.history,
            precision=args.precision,
            remove_wcs=args.remove_wcs,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
    
    if args.verbose:
        print(f"\nSuccessfully processed {success_count}/{len(files)} files")
    
    return 0 if success_count == len(files) else 1


if __name__ == '__main__':
    sys.exit(main())
