import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import argparse
import multiprocessing as mp
from functools import partial

def process_single_file(args):
    """Process a single FITS file and return its coverage mask."""
    fname, ref_wcs, ref_shape = args
    with fits.open(fname) as hdul:
        img_wcs = WCS(hdul[0].header)
        img_shape = hdul[0].data.shape
        return create_coverage_mask(ref_wcs, img_wcs, ref_shape, img_shape)

def create_coverage_mask(ref_wcs, img_wcs, shape, img_shape):
    """Create a coverage mask by checking if reference points fall within image footprint."""
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    world_coords = ref_wcs.pixel_to_world(x, y)
    img_pixels = img_wcs.world_to_pixel(world_coords)
    mask = (img_pixels[0] >= 0) & (img_pixels[0] < img_shape[1]) & \
           (img_pixels[1] >= 0) & (img_pixels[1] < img_shape[0])
    return mask.astype(int)

def create_reference_wcs(input_files, sampling_factor=8, scale_factor=1.5):
    """Create a reference TAN projection WCS based on input images.
    
    Args:
        input_files: List of input FITS files
        sampling_factor: Factor by which to increase pixel size
        scale_factor: Factor by which to increase field size
    
    Returns:
        WCS object with TAN projection and shape tuple
    """
    centers = []
    scales = []
    fields = []
    
    for fname in input_files:
        with fits.open(fname) as hdul:
            w = WCS(hdul[0].header)
            ny, nx = hdul[0].data.shape

            center_x, center_y = nx/2, ny/2
            rd = w.all_pix2world([[center_x, center_y], [0, 0], [center_x, center_y+1]], 0)
            
            center = SkyCoord(rd[0][0]*u.deg, rd[0][1]*u.deg, frame='fk5')
            corner = SkyCoord(rd[1][0]*u.deg, rd[1][1]*u.deg, frame='fk5')
            pixel_point = SkyCoord(rd[2][0]*u.deg, rd[2][1]*u.deg, frame='fk5')
            
            field_size = center.separation(corner) * 2
            pixel_size = pixel_point.separation(center)
            
            centers.append(center)
            scales.append(pixel_size.deg)
            fields.append(field_size.deg)
    
    ra = np.mean([c.ra.deg for c in centers])
    dec = np.mean([c.dec.deg for c in centers])
    pixel_scale = np.max(scales) * sampling_factor
    
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [pixel_scale, pixel_scale]
    
    base_size_deg = np.median(fields)
    pixel_scale = np.median(scales) * sampling_factor
    
    ra_min = np.min([c.ra.deg for c in centers])
    ra_max = np.max([c.ra.deg for c in centers])
    dec_min = np.min([c.dec.deg for c in centers])
    dec_max = np.max([c.dec.deg for c in centers])
    
    mean_dec = np.radians(np.mean([dec_min, dec_max]))
    ra_spread = (ra_max - ra_min) * np.cos(mean_dec)
    dec_spread = dec_max - dec_min
    
    total_ra_size = (base_size_deg + ra_spread)
    total_dec_size = (base_size_deg + dec_spread) 

    size_ra = int(total_ra_size / pixel_scale)
    size_dec = int(total_dec_size / pixel_scale)
    
    size_ra = max(64, size_ra)
    size_dec = max(64, size_dec)
    
    w.wcs.crpix = [size_ra/2, size_dec/2]
    
    return w, (size_dec, size_ra)

def find_maximal_rectangle(matrix, mode='in'):
    """Find the largest rectangular region in a matrix."""
    if mode == 'out':
        working_matrix = matrix > 0
    else:
        working_matrix = matrix == np.max(matrix)
    
    if not working_matrix.any():
        return (0, 0, 0, 0)
    
    rows, cols = working_matrix.shape
    height = np.zeros((rows + 1, cols), dtype=int)
    max_area = 0
    max_rect = (0, 0, 0, 0)
    
    for i in range(rows):
        for j in range(cols):
            if working_matrix[i, j]:
                height[i + 1, j] = height[i, j] + 1
            else:
                height[i + 1, j] = 0
        
        stack = []
        j = 0
        while j < cols:
            start = j
            while stack and height[i + 1, stack[-1]] > height[i + 1, j]:
                h = height[i + 1, stack.pop()]
                w = j - stack[-1] - 1 if stack else j
                area = h * w
                if area > max_area:
                    if mode == 'inside':
                        rect_slice = matrix[max(0, i + 1 - h):i + 1, max(0, j - w):j]
                        if np.all(rect_slice > 0):
                            max_area = area
                            max_rect = (start - w, i + 1 - h, w, h)
                    else:
                        max_area = area
                        max_rect = (start - w, i + 1 - h, w, h)
            stack.append(j)
            j += 1
        
        while stack:
            h = height[i + 1, stack.pop()]
            w = cols - stack[-1] - 1 if stack else cols
            area = h * w
            if area > max_area:
                if mode == 'inside':
                    rect_slice = matrix[max(0, i + 1 - h):i + 1, max(0, cols - w):cols]
                    if np.all(rect_slice > 0):
                        max_area = area
                        max_rect = (cols - w, i + 1 - h, w, h)
                else:
                    max_area = area
                    max_rect = (cols - w, i + 1 - h, w, h)
    
    return max_rect

def process_single_file(args):
    """Process a single FITS file and return its coverage mask.
    
    Args:
        args: tuple containing (filename, ref_wcs, ref_shape)
    Returns:
        numpy array containing the coverage mask
    """
    fname, ref_wcs, ref_shape = args
    with fits.open(fname) as hdul:
        img_wcs = WCS(hdul[0].header)
        img_shape = hdul[0].data.shape
        return create_coverage_mask(ref_wcs, img_wcs, ref_shape, img_shape)

def create_coverage_mask(ref_wcs, img_wcs, shape, img_shape):
    """Create a coverage mask by checking if reference points fall within image footprint."""
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    world_coords = ref_wcs.pixel_to_world(x, y)
    img_pixels = img_wcs.world_to_pixel(world_coords)
    mask = (img_pixels[0] >= 0) & (img_pixels[0] < img_shape[1]) & \
           (img_pixels[1] >= 0) & (img_pixels[1] < img_shape[0])
    return mask.astype(int)


def find_optimal_frame(fits_files, sampling_factor=8, scale_factor=1.5, zoom_factor=1.0, num_processes=None, mode='in'):
    """
    Find the optimal WCS frame for a set of FITS images.
    
    Args:
        fits_files: List of input FITS files
        sampling_factor: Factor for initial sampling (default: 8)
        scale_factor: Factor for field scaling (default: 1.5)
        zoom_factor: Additional zoom factor for final output (default: 1.0)
        num_processes: Number of processes for parallel processing (default: CPU count)
        
    Returns:
        dict: FITS header keywords
    """
    # Create reference WCS
    ref_wcs, ref_shape = create_reference_wcs(
        fits_files,
        sampling_factor=sampling_factor,
        scale_factor=scale_factor
    )
    
    # Initialize coverage mask
    coverage = np.zeros(ref_shape, dtype=int)
    
    # Process images in parallel
    num_processes = num_processes or mp.cpu_count()
    print(f"Processing {len(fits_files)} files using {num_processes} processes...")
    
    # Create pool of workers
    with mp.Pool(num_processes) as pool:
        # Prepare arguments for each file
        process_args = [(fname, ref_wcs, ref_shape) for fname in fits_files]
        
        # Process files in parallel and collect results
        masks = pool.map(process_single_file, process_args)
        
        # Sum up all masks
        coverage = sum(masks)

    # Find maximal rectangle
    max_rect = find_maximal_rectangle(coverage, mode=mode)
    x, y, w, h = max_rect
    
    # Create output header
    header = ref_wcs.to_header()
    header['NAXIS'] = 2
    header['NAXIS1'] = w * sampling_factor * zoom_factor
    header['NAXIS2'] = h * sampling_factor * zoom_factor
    header['BITPIX'] = -32 # 32 bit float
    header['CRPIX1'] -= x
    header['CRPIX2'] -= y
    
    # Scale pixel size
    final_scale = sampling_factor * zoom_factor
    header['CDELT1'] /= final_scale
    header['CDELT2'] /= final_scale
    header['CRPIX1'] *= final_scale
    header['CRPIX2'] *= final_scale
    
    # Add metadata
    header['SAMPLING'] = sampling_factor
    header['SCALE'] = scale_factor
    header['NIMAGES'] = len(fits_files)
    
    return header

def write_header(header, filename):
    """Write FITS header to a text file."""
    with open(filename, 'w') as f:
        for card in header.cards:
            f.write(card.image + '\n')

def main():
    parser = argparse.ArgumentParser(description='Find overlapping regions in FITS images')
    parser.add_argument('files', nargs='+', help='Input FITS files')
    parser.add_argument('--sampling-factor', type=int, default=8,
                       help='Sampling factor for WCS calculations')
    parser.add_argument('--scale-factor', type=float, default=1.5,
                       help='Factor by which to scale the field size')
    parser.add_argument('--zoom-factor', type=float, default=1.0,
                       help='Additional zoom factor for final output')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of parallel processes to use')
    parser.add_argument('--mode', choices=['in', 'out'], default='in',
                       help='Output inner (all images) or outer (partial coverage) rectangle')
    parser.add_argument('--header', default='skel.hdr',
                       help='Output header file')
#    parser.add_argument('--output-wcs', default=None,
#                       help='Optional output FITS file containing reference WCS')
#    parser.add_argument('--min-images', type=int, default=None,
#                       help='Minimum number of images required for coverage')
#    parser.add_argument('--overlap-cutout', default=None,
#                       help='Optional output FITS file for overlap cutout')
    
    args = parser.parse_args()
    
    try:
        header = find_optimal_frame(
            args.files,
            sampling_factor=args.sampling_factor,
            scale_factor=args.scale_factor,
            zoom_factor=args.zoom_factor,
            num_processes=args.num_processes,
            mode=args.mode
        )
        write_header(header, args.header)
        print(f"Written WCS header to {args.header}")

    # Optionally write FITS files
#    if args.output_wcs:
#        fits.writeto(args.output_wcs, coverage, header, overwrite=True)
#        print(f"Written reference WCS and coverage map to {args.output_wcs}")
#    
#    if args.overlap_cutout:
#        cutout_data = coverage[y:y+h, x:x+w]
#        fits.writeto(args.overlap_cutout, cutout_data, header, overwrite=True)
#        print(f"Written overlap cutout to {args.overlap_cutout}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())

