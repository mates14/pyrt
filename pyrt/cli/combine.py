#!/usr/bin/python3
"""
combine-images: Intelligent FITS image combination with photometric weighting

This tool combines multiple FITS images using sophisticated weighting based on
photometric calibration, with automatic outlier rejection and optimal WCS frame
calculation.

Usage Modes:
-----------

5. Pre-projected mode (-P):
   For frames already on the same WCS grid with WGHT scalar weights in their
   headers and per-pixel weight maps referenced via WGHTFILE. Skips selection,
   frame calculation, weight computation, and reprojection — feeds frames
   directly to mAdd and always produces combinedw.fits alongside combined.fits.

   Example:
       combine-images -P --skel tile.hdr -o combined.fits frame*t.fits

1. Automatic weighted mode (default):
   Performs outlier rejection, calculates optimal WCS frame, computes S/N-based
   weights, and combines images. Requires MAGZERO keyword in FITS headers.

   Example:
       combine-images input*.fits
       combine-images -o output.fits input*.fits

2. Uniform weighting mode:
   Equal weight for all images. No photometric calibration required.

   Example:
       combine-images -u input*.fits

3. Manual mode (expert control):
   Skip automatic steps when you have pre-selected frames or a specific target
   WCS (useful for inter-instrument subtraction, precise alignment, etc.)

   Example:
       combine-images --skeleton custom.hdr --no-selection input*.fits

4. Skeleton-only mode:
   Calculate optimal WCS frame and write skeleton header without combining.

   Example:
       combine-images --skeleton-only skel.hdr input*.fits

Key Features:
-------------
- **Intelligent outlier rejection**: Spatial clustering to remove misaligned frames
- **Optimal WCS calculation**: Finds the best-fit coordinate frame covering all inputs
- **Photometric weighting**: Signal-to-noise optimization for best combined S/N
- **Uniform weighting**: Optional equal weighting for uncalibrated data
- **Smart MAGZERO handling**: Skips files without calibration, suggests --uniform
- **Time-variable source support**: GRB mode with power-law brightness evolution
- **Photometry file support**: Use measured magnitudes from dophot output for optimal SNR weighting
- **Weight map handling**: Supports per-pixel weighting (vignetting correction)
- **Parallel processing**: Multi-core image projection and reprojection
- **Characteristic time**: Proper time tagging for transient observations
- **Metadata preservation**: Propagates observing conditions to output

Weighting Algorithms:
--------------------

Photometric weighting (default):
  For each image, calculates optimal weight based on:
  - Expected source brightness (from MAGZERO and exposure time)
  - Background noise (measured from image statistics)
  - Signal-to-noise contribution to final combination

  Files without MAGZERO are skipped and reported. If all files lack MAGZERO,
  the tool suggests using --uniform mode.

  For time-variable sources (GRB mode), adjusts expected brightness using
  power-law decay model: mag(t) = mag0 + decay * log10(t/t0)

  For measured photometry (--photfile mode), uses actual magnitudes from a
  dophot-style output file. This is useful for combining images to maximize
  SNR for precise astrometric position determination.

Uniform weighting (--uniform):
  All images receive equal weight. Useful for:
  - Uncalibrated data without MAGZERO
  - Quick combinations without optimization
  - Testing and debugging

Spatial weighting:
  Per-pixel weight maps (WGHTFILE keyword) are combined with scalar weights
  to handle vignetting, bad pixel masks, or other spatially-varying effects.

Technical Details:
------------------
- Uses Montage toolkit (mProjectPP/mProjectPX) for accurate reprojection
- Handles complex projections via mProjectPX (ZPN → TAN+SIP conversion)
- Parallel processing for large image sets
- Proper error propagation (GAIN, RNOISE)
- Characteristic time calculation for transients
- Safe handling of missing calibration data
"""

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import argparse
from pathlib import Path
import subprocess
import tempfile
import multiprocessing as mp
import datetime
from tqdm import tqdm

# This is to silence a particular annoying warning (MJD not present in a fits file)
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

# Keywords to copy from input files to output
COPY_KEYWORDS = [
    'CTIME', 'USEC', 'JD', 'DATE-OBS',
    'TARGET', 'TARSEL', 'TARTYPE', 'OBSID', 'IMGID', 'PROC', 'CCD_NAME',
    'MOONDIST', 'MOONRA', 'MOONDEC', 'MOONPHA', 'MOONALT', 'MOONAZ',
    'EXPOSURE', 'EXPTIME', 'INSTRUME', 'TELESCOP', 'ORIGIN', 'FOC_NAME',
    'SCRIPREP', 'SCRIPT', 'SCR_COMM', 'COMM_NUM', 'CCD_TYPE', 'CCD_SER', 'CCD_CHIP',
    'IMAGETYP', 'OBJECT', 'SLITPOSX', 'SLITPOSY',
    'BINNING', 'BINX', 'BINY', 'CCD_TEMP', 'COOLING', 'CCD_SET', 'SHUTTER',
    'CHAN', 'CHAN1', 'FTSHUT', 'ACQMODE', 'ACCCYCLE', 'ACCNUM', 'KINCYCLE', 'KINNUM',
    'IMGFREQ', 'ACQTIME', 'USEFT', 'FILTCR', 'ADCMODE', 'EMON', 'ADCHANEL', 'OUTAMP',
    'HSPEED', 'PREAMP', 'GAIN', 'VSPEED', 'SAMPLI', 'EMADV', 'GAINMODE', 'BASECLAM',
    'FAN', 'FILTA',
    'MNT_NAME', 'LATITUDE', 'LONGITUD', 'ALTITUDE', 'AMBTEMP', 'DUT1',
    'ORIRA', 'ORIDEC', 'OEPOCH', 'PMRA', 'PMDEC', 'OFFSRA', 'OFFSDEC', 'DRATERA', 'DRATEDEC',
    'TRACKING', 'SKYSPDRA', 'OBJRA', 'OBJDEC', 'TARRA', 'TARDEC', 'CORR_RA', 'CORR_DEC',
    'TELRA', 'TELDEC', 'JD_HELIO', 'U_TELRA', 'U_TELDEC', 'TEL_ALT', 'TEL_AZ', 'PARKTIME',
    'AIRMASS', 'HA', 'LST', 'MOVE_NUM', 'CORR_IMG', 'MNT_ROTA', 'MNT_FLIP',
    'IH', 'ID', 'NP', 'CH', 'ME', 'MA', 'HDCD', 'HDSD', 'HHSH', 'HHCH', 'HDSH', 'HDCH',
    'RA_TPOS', 'RA_APOS', 'RA_FAULT', 'DEC_TPOS', 'DEC_APOS', 'MNT_INFO',
    'SUN_ALT', 'SUN_AZ', 'CAM_FILT', 'AVERAGE', 'STDEV', 'FILTER',
    'RA_ERR', 'DEC_ERR', 'POS_ERR', 'FWHM', 'ELLIP', 'BGSIGMA', 'MAGZERO', 'DMAGZERO'
]

# ============================================================================
# WCS Frame Calculation Functions (from fits_overlap)
# ============================================================================

def process_single_file_coverage(args):
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

def create_reference_wcs(input_files, sampling_factor=8, scale_factor=1.5):
    """Create a reference TAN projection WCS based on input images.

    Args:
        input_files: List of input FITS files
        sampling_factor: Factor by which to increase pixel size
        scale_factor: Factor by which to increase field size

    Returns:
        WCS object with TAN projection and shape tuple
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

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

def find_optimal_frame(fits_files, sampling_factor=8, scale_factor=1.5, zoom_factor=1.0, num_processes=None, mode='in'):
    """
    Find the optimal WCS frame for a set of FITS images.

    Args:
        fits_files: List of input FITS files
        sampling_factor: Factor for initial sampling (default: 8)
        scale_factor: Factor for field scaling (default: 1.5)
        zoom_factor: Additional zoom factor for final output (default: 1.0)
        num_processes: Number of processes for parallel processing (default: CPU count)
        mode: 'in' for full coverage or 'out' for partial coverage (default: 'in')

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
        masks = pool.map(process_single_file_coverage, process_args)

        # Sum up all masks
        coverage = sum(masks)

    # Find maximal rectangle
    max_rect = find_maximal_rectangle(coverage, mode=mode)
    x, y, w, h = max_rect

    # Create output header
    header = ref_wcs.to_header()
    header['NAXIS'] = 2
    header['NAXIS1'] = int(w * sampling_factor * zoom_factor)
    header['NAXIS2'] = int(h * sampling_factor * zoom_factor)
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

# ============================================================================
# Selection and Frame Calculation Functions (from combine)
# ============================================================================

def get_image_center(header):
    """Get the center position of an image in RA/Dec.

    Args:
        header: FITS header with WCS information

    Returns:
        tuple: (ra, dec) in degrees for image center
    """
    wcs = WCS(header)
    naxis1 = header['NAXIS1']
    naxis2 = header['NAXIS2']

    # Get center pixel coordinates
    center_x = naxis1 / 2
    center_y = naxis2 / 2

    # Convert to world coordinates
    center_world = wcs.pixel_to_world(center_x, center_y)
    return center_world.ra.deg, center_world.dec.deg

def find_valid_images_cluster(fits_files):
    """Find valid images using hierarchical clustering based on spatial proximity.

    Automatically determines outliers by analyzing the distribution of image centers.
    Images that are spatially isolated (no nearby neighbors) are rejected as outliers.

    The threshold for "nearby" is determined from the typical image size and the
    natural gaps in the distribution of nearest-neighbor distances.

    Args:
        fits_files: List of FITS file paths

    Returns:
        tuple: (valid_files, rejected_files) - lists of file paths
    """
    centers = []
    scales = []
    all_headers = []

    # Collect centers and pixel scales
    for fits_file in fits_files:
        try:
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                center = get_image_center(header)

                # Get pixel scale (in degrees)
                if 'CDELT1' in header:
                    scale = abs(header['CDELT1'])
                else:
                    cd = np.array([[header.get('CD1_1', 0), header.get('CD1_2', 0)],
                                 [header.get('CD2_1', 0), header.get('CD2_2', 0)]])
                    scale = np.sqrt(np.abs(np.linalg.det(cd)))

                # Get image size in degrees (diagonal)
                naxis1 = header['NAXIS1']
                naxis2 = header['NAXIS2']
                size_deg = np.sqrt((naxis1 * scale)**2 + (naxis2 * scale)**2)

                centers.append(center)
                scales.append(size_deg)
                all_headers.append((fits_file, header))
        except Exception as e:
            print(f"Warning: Could not process {fits_file}: {str(e)}")
            continue

    if not centers:
        raise ValueError("No valid images found")

    centers = np.array(centers)
    scales = np.array(scales)

    # Calculate all pairwise distances
    n_images = len(centers)

    # Handle single image case - no clustering needed
    if n_images == 1:
        return [all_headers[0][0]], []

    distances = np.zeros((n_images, n_images))

    for i in range(n_images):
        for j in range(i+1, n_images):
            # Use cos(dec) correction for RA distances
            cos_dec = np.cos(np.radians(centers[i,1]))
            dist = np.sqrt(
                ((centers[i,0] - centers[j,0]) * cos_dec)**2 +
                (centers[i,1] - centers[j,1])**2
            )
            distances[i,j] = dist
            distances[j,i] = dist

    # For each point, get its distances to other points
    sorted_distances = np.sort(distances, axis=1)

    # The first column (index 0) will be zeros (self-distances)
    # Look at the distribution of nearest neighbor distances (index 1)
    nearest_distances = sorted_distances[:,1]

    # Get typical image size
    typical_size = np.median(scales)

    # Find natural break in nearest neighbor distances
    sorted_nn = np.sort(nearest_distances)
    gaps = sorted_nn[1:] - sorted_nn[:-1]

    # Find significant gaps (larger than 1/10th typical image size)
    significant_gaps = np.where(gaps > typical_size/10)[0]

    if len(significant_gaps) > 0:
        # Use the largest significant gap as the break point
        threshold = sorted_nn[significant_gaps[0]]
    else:
        # If no significant gaps, use 2x typical image size
        threshold = 2 * typical_size

    # A point is valid if it has at least one neighbor closer than the threshold
    valid_mask = nearest_distances < threshold

    valid_files = []
    rejected_files = []

    # Separate valid and rejected files
    for i, (fits_file, _) in enumerate(all_headers):
        if valid_mask[i]:
            valid_files.append(fits_file)
        else:
            rejected_files.append(fits_file)
            print(f"Rejecting {fits_file} as outlier:")
            print(f"  Distance to nearest neighbor: {nearest_distances[i]:.3f} deg")
            print(f"  (Threshold: {threshold:.3f} deg)")

    # Calculate center of valid points
    valid_centers = centers[valid_mask]
    ra_center = np.mean(valid_centers[:,0])
    dec_center = np.mean(valid_centers[:,1])

    # Print summary
    print(f"\nSelection Summary:")
    print(f"Total images: {len(centers)}")
    print(f"Valid images: {len(valid_files)}")
    print(f"Rejected images: {len(rejected_files)}")
    print(f"Center position: RA={ra_center:.3f}, Dec={dec_center:.3f}")
    print(f"Threshold distance: {threshold:.3f} deg")
    print(f"Typical image size: {typical_size:.3f} deg")

    return valid_files, rejected_files

def write_montage_header(header, output_file='skel.hdr'):
    """Write a FITS header to a Montage-compatible ASCII header file.

    Args:
        header: Dictionary of FITS header keywords and values
        output_file: Output filename (default: skel.hdr)
    """
    with open(output_file, 'w') as f:
        for key, value in header.items():
            if isinstance(value, str):
                f.write(f'{key:8s}= \'{value}\'\n')
            elif isinstance(value, (int, float)):
                f.write(f'{key:8s}= {value}\n')
            else:
                f.write(f'{key:8s}= {value}\n')
    print(f"Written skeleton header to {output_file}")


def fits_to_skeleton(fits_path, hdr_path='skel.hdr'):
    """Extract WCS header from a FITS file into a Montage-compatible .hdr file.

    Uses mGetHdr from the Montage toolkit, which is the same tool used
    internally when building skeleton headers during frame estimation.

    Args:
        fits_path: Path to the input FITS file
        hdr_path: Output .hdr file path (default: skel.hdr)

    Returns:
        hdr_path
    """
    result = subprocess.run(
        ["mGetHdr", fits_path, hdr_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"mGetHdr failed for {fits_path}: {result.stderr.strip()}")
    print(f"Extracted WCS header from FITS file {fits_path} -> {hdr_path}")
    return hdr_path


def resolve_skeleton(skeleton_arg):
    """Resolve a --skeleton/--skel argument.

    If the argument points to a FITS file (.fits/.fit/.fts), its WCS header
    is extracted with mGetHdr and written to skel.hdr, which is returned.
    Otherwise the argument is returned unchanged.

    Args:
        skeleton_arg: Path passed to --skeleton/--skel, or None

    Returns:
        Path to a Montage-compatible .hdr file, or None
    """
    if skeleton_arg is None:
        return None
    if skeleton_arg.lower().endswith(('.fits', '.fit', '.fts')):
        if not os.path.exists(skeleton_arg):
            raise FileNotFoundError(f"Skeleton FITS file not found: {skeleton_arg}")
        return fits_to_skeleton(skeleton_arg, 'skel.hdr')
    return skeleton_arg


# ============================================================================
# Weighting and Combination Functions (from combine2w)
# ============================================================================

def calculate_chartime(times, weights, t0, rate):
    """Calculate characteristic time of the combined image.

    For time-variable sources, computes the effective time that represents
    the combined observation. Uses sophisticated integral calculation when
    rate information is available, otherwise falls back to weighted average.

    Args:
        times: List of dicts with 'time', 'abs_time', 'exptime' keys
        weights: List of image weights
        t0: Reference time (JD) for power-law calculation
        rate: Magnitude change rate (mags/day) for power-law

    Returns:
        float: Characteristic time in JD, or None if calculation fails
    """
    # If we have t0 and rate, try the sophisticated method first
    if t0 is not None and rate is not None:
        try:
            A2 = rate  # Magnitude change rate
            total_up = 0
            total_down = 0

            for time, weight in zip(times, weights):
                exp = time['exptime']  # in days
                t = time['time']  # in days since T0

                # Skip invalid times
                if t <= 0:
                    continue

                # Calculate components for characteristic time
                up = (np.power(t + exp/2, -A2 + 2) - np.power(t - exp/2, -A2 + 2)) * \
                     np.power(np.power(t + exp/2, -A2 + 1) - np.power(t - exp/2, -A2 + 1), 2) * weight

                down = np.power(np.power(t + exp/2, -A2 + 1) - np.power(t - exp/2, -A2 + 1), 3) * weight

                total_up += up
                total_down += down

            if total_down != 0:
                chartime = (-A2 + 1)/(-A2 + 2) * total_up / total_down
                return chartime + t0
        except Exception as e:
            print(f"Warning: Complex characteristic time calculation failed: {str(e)}")

    # Fallback to simple weighted average of mid-exposure times
    try:
        total_weight = 0
        weighted_time = 0

        for time, weight in zip(times, weights):
            abs_time = time['abs_time']  # Actual JD
            weighted_time += abs_time * weight
            total_weight += weight

        if total_weight > 0:
            print(f"Simple chartime = {weighted_time / total_weight}")
            return weighted_time / total_weight
    except Exception as e:
        print(f"Warning: Simple characteristic time calculation failed: {str(e)}")

    return None

def update_output_header(output_file, inputs, weights, args):
    """Update output FITS file header with combined metadata.

    Propagates relevant keywords from input files, calculates proper exposure time,
    characteristic time, and effective GAIN/RNOISE for the combination.

    Args:
        output_file: Path to output FITS file (will be modified in place)
        inputs: List of input file paths
        weights: Dictionary mapping file paths to weights
        args: Parsed command-line arguments
    """
    times = []
    exposure_times = []

    # Collect metadata from input files
    keyword_values = {key: [] for key in COPY_KEYWORDS}

    for f in inputs:
        if weights[f] <= 0:
            continue

        with fits.open(f) as hdul:
            header = hdul[0].header

            # Collect times for CHARTIME and EXPTIME calculation
            # Try EXPTIME first, fall back to EXPOSURE
            exptime_key = None
            if 'EXPTIME' in header:
                exptime_key = 'EXPTIME'
            elif 'EXPOSURE' in header:
                exptime_key = 'EXPOSURE'

            if exptime_key and weights[f] > 0:
                exptime_sec = float(header[exptime_key])
                exptime_days = exptime_sec / 86400.0

                # Get start time (shutter open) - using CTIME/USEC directly
                try:
                    usec = float(header.get('USEC', 0))
                    sec = float(header.get('CTIME', 0))
                    exp = float(header.get(exptime_key, 0))
                    sec = sec + usec/1e6
                    exp = exp / 86400.0

                    start_time_jd = 2440587.5 + sec/86400.0  # Start of exposure in JD
                    end_time_jd = start_time_jd + exptime_days

                    exposure_times.append({
                        'start': start_time_jd,
                        'end': end_time_jd,
                        'exptime_sec': exptime_sec
                    })

                    # Mid-exposure time for CHARTIME calculation
                    mid_time = start_time_jd + exptime_days/2.0
                    if args.t0:
                        time_since_t0 = mid_time - args.t0
                        times.append({
                            'time': time_since_t0,
                            'abs_time': mid_time,
                            'exptime': exptime_days,
                        })
                    else:
                        times.append({
                            'abs_time': mid_time,
                            'exptime': exptime_days,
                        })
                except (KeyError, ValueError) as e:
                    print(f"Warning: Error processing times for {f}: {e}")
                    continue

            # Collect values for keywords to copy
            for key in COPY_KEYWORDS:
                if key in header:
                    keyword_values[key].append(header[key])

    # Update output file header
    with fits.open(output_file, mode='update') as hdul:
        header = hdul[0].header

        # Calculate proper EXPTIME as last-shutter-close - first-shutter-open
        if exposure_times:
            first_shutter_open = min(exp['start'] for exp in exposure_times)
            last_shutter_close = max(exp['end'] for exp in exposure_times)
            total_exptime_sec = (last_shutter_close - first_shutter_open) * 86400.0
            header['EXPTIME'] = total_exptime_sec
            header['EXPSTART'] = (first_shutter_open, 'First shutter open time (JD)')
            header['EXPEND'] = (last_shutter_close, 'Last shutter close time (JD)')
        else:
            # Fallback to summing individual exposure times
            total_exptime_sec = sum(exp['exptime_sec'] for exp in exposure_times)
            header['EXPTIME'] = total_exptime_sec

        # Calculate and add CHARTIME if applicable
        if times:
            chartime = calculate_chartime(times, [weights[f] for f in inputs if weights[f] > 0],
                                       args.t0, args.rate)
            if chartime is not None:
                header['CHARTIME'] = (chartime, 'Characteristic time of combined image')
        else:
            print("no times -> no chartime")

        # Calculate effective GAIN and RNOISE for combined image
        gain_values = []
        rnoise_values = []
        weight_sum = 0

        for f in inputs:
            if weights[f] <= 0:
                continue

            with fits.open(f) as hdul:
                hdr = hdul[0].header
                gain = float(hdr.get('GAIN', args.gain))
                if 'RNOISE' in hdr:
                    rnoise = float(hdr['RNOISE'])
                elif 'READNOIS' in hdr:
                    rnoise = float(hdr['READNOIS'])
                else:
                    rnoise = np.sqrt(gain * 3.0)  # rough estimate

                weight = weights[f]
                gain_values.append(gain * weight)
                rnoise_values.append((rnoise / gain)**2 * weight)
                weight_sum += weight

        if weight_sum > 0:
            effective_gain = sum(gain_values) / weight_sum
            effective_rnoise = np.sqrt(sum(rnoise_values) / weight_sum) * effective_gain

            header['GAIN'] = (effective_gain, 'Effective gain of combined image')
            header['RNOISE'] = (effective_rnoise, 'Effective read noise of combined image')

        # Copy most common value for each keyword
        from collections import Counter
        for key in COPY_KEYWORDS:
            values = keyword_values[key]
            if values:
                if key in ['COMMENT', 'HISTORY']:
                    for value in set(values):
                        header[key] = value
                elif key in ['GAIN', 'RNOISE', 'EXPTIME', 'EXPOSURE']:
                    continue  # Already calculated above
                else:
                    most_common = Counter(values).most_common(1)[0][0]
                    header[key] = most_common

        # Update combination metadata
        header['NCOMBINE'] = (len([w for w in weights.values() if w > 0]),
                            'Number of images combined')
        if args.t0 is not None:
            header['T0'] = (args.t0, 'Reference time for magnitude calculation')
        if args.rate is not None:
            header['MAGRATE'] = (args.rate, 'Magnitude change rate (mag/day)')
        header['MAG0'] = (args.brightness, 'Reference magnitude')

        # Add processing information
        header['COMBINER'] = ('combine-images', 'Tool used for combining')
        header['DATE'] = (datetime.datetime.utcnow().isoformat(),
                        'UTC date when file was combined')

        hdul.flush()

def calculate_background_stats(data):
    """Calculate background sigma using row differences method.

    Computes background noise robustly by looking at row-to-row differences,
    which is insensitive to large-scale gradients and objects.

    Args:
        data: 2D numpy array of image data

    Returns:
        tuple: (sigma, median) - background standard deviation and median
    """
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

    scale_factor = 1.0489  # Conversion from median difference to standard deviation
    sigma = np.nanmedian(ndiff[:j])
    median = np.nanmedian(data[~np.isnan(data)])

    return scale_factor * sigma, median

def parse_photometry_file(photfile, input_files):
    """Parse photometry file with measured magnitudes.

    Reads a dophot-style output file and extracts magnitudes for each input file.
    The file must contain entries for ALL input files.

    Args:
        photfile: Path to photometry file
        input_files: List of input FITS file paths

    Returns:
        dict: Mapping from filename to (magnitude, error) tuple

    Raises:
        ValueError: If any input file is missing from the photometry file
    """
    photometry = {}

    with open(photfile, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            filename = parts[0]
            try:
                mag = float(parts[11])  # Column 12 (0-indexed: 11)
                magerr = float(parts[12])  # Column 13 (0-indexed: 12)
                # Skip invalid magnitudes (99.999 typically means not found)
                if mag > 90:
                    continue
                photometry[filename] = (mag, magerr)
            except ValueError:
                continue

    # Check that all input files have photometry
    input_basenames = {os.path.basename(f): f for f in input_files}
    missing = []
    result = {}

    for basename, fullpath in input_basenames.items():
        if basename in photometry:
            result[fullpath] = photometry[basename]
        else:
            missing.append(basename)

    if missing:
        raise ValueError(
            f"Photometry file {photfile} missing entries for {len(missing)} file(s):\n"
            + "\n".join(f"  - {f}" for f in missing[:10])
            + (f"\n  ... and {len(missing)-10} more" if len(missing) > 10 else "")
        )

    return result


def get_image_time(header):
    """Get image mid-exposure time in JD from FITS header.

    Args:
        header: FITS header with CTIME, USEC, and EXPTIME (or EXPOSURE) keywords

    Returns:
        float: Mid-exposure time in Julian Date
    """
    usec = float(header.get('USEC', 0))
    sec = float(header.get('CTIME',0)) + usec/1e6

    # Try EXPTIME first, fall back to EXPOSURE (both in seconds)
    if 'EXPTIME' in header:
        exp_sec = float(header['EXPTIME'])
    elif 'EXPOSURE' in header:
        exp_sec = float(header['EXPOSURE'])
    else:
        raise ValueError("Neither EXPTIME nor EXPOSURE keyword found in header")

    exp = exp_sec / 86400.0  # Convert to days
    return 2440587.5 + sec/86400.0 + exp/2.0  # Mid-exposure time in JD

def calculate_brightness(time, t0, mag0, rate):
    """Calculate expected source brightness at given time.

    For time-variable sources, applies power-law decay model.
    For static sources, returns constant brightness.

    Args:
        time: Observation time (JD)
        t0: Reference time (JD)
        mag0: Reference magnitude (at 1 day after t0)
        rate: Decay rate (mags per decade in time)

    Returns:
        float: Expected magnitude at the given time
    """
    if t0 is not None and rate is not None:
        dt = (time - t0) * 86400.0  # Convert to seconds
        if dt <= 0:
            raise ValueError(f"Invalid time difference: {dt/86400.0:.2f} days before t0")
        # mag0 is magnitude at 1 day after t0
        return mag0 + rate * 2.5 * np.log10(dt/86400.0)
    else:
        return mag0

def compute_weights(files, gain, t0, mag0, rate, uniform=False, photometry=None, quiet=False):
    """Compute optimal weights for image combination.

    In weighted mode (default), uses photometric calibration (MAGZERO) and
    background noise to calculate the optimal weight for each image based on S/N.
    Files without MAGZERO are skipped.

    In uniform mode, assigns equal weight to all images (no MAGZERO required).

    Args:
        files: List of FITS file paths
        gain: CCD gain (e-/ADU)
        t0: Reference time for variable sources (JD)
        mag0: Reference magnitude
        rate: Brightness change rate (mags/day)
        uniform: If True, use uniform weighting (default: False)
        photometry: Optional dict mapping file paths to (mag, magerr) tuples.
                   When provided, uses actual measured magnitudes instead of
                   calculating from time and power-law model.

    Returns:
        tuple: (weights_dict, skipped_files_list)
            weights_dict: Mapping from file path to weight value
            skipped_files_list: List of files skipped due to missing MAGZERO
    """
    weights = {}
    skipped = []

    if uniform:
        # Uniform weighting mode: assign weight 1.0 to all files
        if not quiet:
            print("Using uniform weighting (all images weighted equally)")
        for f in tqdm(files, desc="Computing weights", unit="img", disable=not quiet):
            weights[f] = 1.0
            # Update FITS header with weight
            result = subprocess.run(
                ["fitsheader", "-w", f"WGHT=1.0", f],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: Failed to update weight in {f}: {result.stderr}")

        n_images = len(files)
        normalized_weights = {f: float(n_images) for f in files}
        if not quiet:
            print(f"Normalized {n_images} images with uniform weight")
        return normalized_weights, skipped

    # Weighted mode: calculate S/N-based weights
    total_counts = 0

    if photometry:
        print("Using measured magnitudes from photometry file")

    # Two-pass weight calculation; pass 1 is ~40x faster than pass 2, so use a
    # single progress bar with total=41*N: pass 1 advances 1 per file, pass 2 advances 40.
    _pbar = tqdm(total=41 * len(files), desc="Computing weights", unit="img",
                 disable=not quiet)

    # First pass - calculate total counts, skip files without MAGZERO
    for f in files:
        with fits.open(f) as hdul:
            header = hdul[0].header
            if 'MAGZERO' not in header:
                print(f"Warning: Skipping {f} - MAGZERO keyword missing")
                skipped.append(f)
                _pbar.update(1)
                continue

            # Get magnitude: from photometry file or calculate from time
            if photometry:
                mag, magerr = photometry[f]
            else:
                time = get_image_time(header)
                mag = calculate_brightness(time, t0, mag0, rate)
            counts = gain * np.power(10, -0.4 * (mag - header['MAGZERO']))

            if counts <= 0:
                print(f"Warning: Skipping {f} - invalid counts: {counts}")
                skipped.append(f)
                _pbar.update(1)
                continue

            total_counts += counts
        _pbar.update(1)

    if total_counts <= 0:
        _pbar.close()
        print("Warning: no MAGZERO found in any input file — falling back to uniform weighting")
        n = len(files)
        return {f: float(n) for f in files}, []

    # Second pass - calculate weights
    for f in files:
        if f in skipped:
            weights[f] = 0.0
            _pbar.update(40)
            continue

        try:
            with fits.open(f) as hdul:
                header = hdul[0].header
                data = hdul[0].data

                if data is None or len(data.shape) != 2:
                    raise ValueError(f"Invalid data array in {f}")

                # Get magnitude: from photometry file or calculate from time
                if photometry:
                    mag, magerr = photometry[f]
                else:
                    time = get_image_time(header)
                    mag = calculate_brightness(time, t0, mag0, rate)
                counts = gain * np.power(10, -0.4 * (mag - header['MAGZERO']))

                # Calculate background stats
                sigma, _ = calculate_background_stats(data)

                # Calculate weight using signal-to-noise optimization
                # Weight each image by its contribution to final S/N
                # Use actual FWHM from header if available, else fall back to 4 px
                fwhm = header.get('FWHM', 4.0)
                denominator = counts + np.pi * (fwhm / 2) ** 2 * (gain * sigma) ** 2
                if denominator <= 0:
                    raise ValueError(f"Invalid weight calculation in {f}")

                weight = np.power(counts/total_counts, 2) / denominator
                weights[f] = weight
                if not quiet:
                    print(f"Weight for {f}: {weight:.6e}")

                # Update FITS header with weight
                result = subprocess.run(
                    ["fitsheader", "-w", f"WGHT={weight:.6g}", f],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to update weight in {f}: {result.stderr}")

        except Exception as e:
            print(f"Warning: Error processing {f}: {str(e)}")
            weights[f] = 0.0
            if f not in skipped:
                skipped.append(f)
        finally:
            _pbar.update(40)

    _pbar.close()

    if all(w == 0 for w in weights.values()):
        raise ValueError("No valid weights calculated")

    total_weights = sum(weights.values())
    if not quiet:
        print(f"Sum of all weights: {total_weights:.6e}")
        print(f"Number of images with weight > 0: {sum(1 for w in weights.values() if w > 0)}")

    # Normalize and scale weights: sum = N (compensates for mAdd's automatic averaging)
    if total_weights > 0:
        n_images = sum(1 for w in weights.values() if w > 0)
        normalized_weights = {f: w * n_images / total_weights for f, w in weights.items()}
        if not quiet:
            print(f"Normalized weights to sum={n_images} (compensates for mAdd averaging)")
        return normalized_weights, skipped
    else:
        return weights, skipped

def process_single_image(input_data):
    """Process a single image and its weight map if present.

    Applies scalar weight, reprojects to target WCS using mProjectPP or mProjectPX,
    and handles associated weight maps for vignetting correction.

    Optionally subtracts the background (via pyrt-phcat -B) and applies synthetic
    flat field correction before combination.

    Args:
        input_data: Tuple of (input_file, scalar_weight, weighted_dir, proj_dir,
                    skel_hdr, drizzle, weights_dir, projw_dir, subtract_background)

    Returns:
        dict: {'image': projected_file, 'weight_map': projected_weight_map or None}
    """
    input_file, scalar_weight, weighted_dir, proj_dir, skel_hdr, drizzle, weights_dir, projw_dir, subtract_background, quiet = input_data

    base_name = Path(input_file).stem
    output_weighted = weighted_dir / f"{base_name}_weighted.fits"
    output_proj = proj_dir / f"{base_name}_proj.fits"

    weight_map_file = None
    weight_map_weighted = None
    weight_map_proj = None

    # Step 1: Background subtraction (optional, -B flag)
    if subtract_background:
        base_path = Path(input_file)
        bgsub_file = base_path.parent / f"{base_path.stem}s.fits"
        cat_file = base_path.parent / f"{base_path.stem}.cat"
        import shutil as _shutil
        phcat_cmd = "pyrt-phcat" if _shutil.which("pyrt-phcat") else f"{sys.executable} -m pyrt.cli.phcat"
        result = subprocess.run(
            [*phcat_cmd.split(), "-B", input_file],
            capture_output=True, text=True
        )
        if result.returncode == 0 and bgsub_file.exists():
            if cat_file.exists():
                cat_file.unlink()
            working_file = str(bgsub_file)
            if not quiet:
                print(f"Background subtracted: {input_file} -> {bgsub_file.name}")
        else:
            bgsub_file = None  # nothing to clean up
            print(f"Warning: Background subtraction failed for {input_file}, using original")
            if result.stderr:
                print(f"  phcat error: {result.stderr.strip()}")
            working_file = input_file
    else:
        bgsub_file = None
        working_file = input_file

    with fits.open(working_file) as hdul:
        # Check for weight map
        has_weight_map = 'WGHTFILE' in hdul[0].header
        if has_weight_map:
            if not quiet:
                print(f"File {hdul[0].header['WGHTFILE']} is a weight map for {working_file}")
            weight_map_file = hdul[0].header['WGHTFILE']
            if not os.path.exists(weight_map_file):
                print(f"Warning: Weight file {weight_map_file} not found")
                has_weight_map = False

        # Steps 2-3: Synthetic flat field correction
        # Flat is named {filebase}f.fits where weight is {filebase}w.fits
        flat_data = None
        if has_weight_map and weight_map_file.endswith('w.fits'):
            flat_file = weight_map_file[:-len('w.fits')] + 'f.fits'
            if os.path.exists(flat_file):
                with fits.open(flat_file) as fhdul:
                    raw_flat = fhdul[0].data.astype(float)
                    flat_median = np.median(raw_flat[raw_flat != 0]) if np.any(raw_flat != 0) else 1.0
                    flat_data = raw_flat / flat_median
                if not quiet:
                    print(f"Applying flat field correction from {flat_file} (median={flat_median:.4g})")
            else:
                if not quiet:
                    print(f"Note: No synthetic flat found for {working_file} (expected {flat_file})")

        # Always use mProjectPX - it's a smart wrapper that:
        # - Calls mProjectPP directly for natively supported projections
        # - Converts to TAN+SIP for complex projections (ZPN, or old-style distortions)
        # This handles the case where CTYPE says TAN but WCS interprets as PLA
        # due to old-style distortion coefficients
        proj_tool = "pyrt-mproject" # "mProjectPX"
        if not quiet:
            print(f"Using {proj_tool}")

        # Apply flat field correction to image data if available
        img_data = hdul[0].data / flat_data if flat_data is not None else hdul[0].data

        # Process weight map if present
        if has_weight_map:
            try:
                weight_map_weighted = weights_dir / f"{base_name}_wmap_weighted.fits"
                weight_map_proj = projw_dir / f"{base_name}_wmap_proj.fits"

                # Copy and scale weight map, including scalar weight so that
                # mAdd's division produces sum(w_i*spatial_i*img_i)/sum(w_i*spatial_i)
                with fits.open(weight_map_file) as whdul:
                    weighted_wmap = whdul[0].data / np.max(whdul[0].data) * scalar_weight
                    new_whdu = fits.PrimaryHDU(weighted_wmap, header=whdul[0].header)
                    new_whdul = fits.HDUList([new_whdu])
                    new_whdul.writeto(weight_map_weighted)

                    weighted_data = img_data * weighted_wmap
            except:
                print("weightmap preparation failed, resetting to no weightmap")
                has_weight_map = False

        if not has_weight_map:
            weighted_data = img_data * scalar_weight

        # Write weighted image
        new_hdu = fits.PrimaryHDU(weighted_data, header=hdul[0].header)
        new_hdul = fits.HDUList([new_hdu])
        new_hdul.writeto(output_weighted)

        # Project the image
        if not quiet:
            print(proj_tool, output_weighted)
        result = subprocess.run([
            proj_tool,
            "-z", str(drizzle),
            str(output_weighted),
            str(output_proj),
            str(skel_hdr)
        ], stdout=subprocess.PIPE if quiet else None, stderr=subprocess.PIPE)

        if result.returncode != 0:
            out = (result.stdout or b'').decode('utf-8', errors='replace') if quiet else ''
            err = (result.stderr or b'').decode('utf-8', errors='replace') if quiet else ''
            raise RuntimeError(
                f"Failed to project image (rc={result.returncode}):\n"
                f"stdout: {out}\nstderr: {err}"
            )

        # Project weight map if present
        if has_weight_map:
            if not quiet:
                print(proj_tool, weight_map_weighted)
            result = subprocess.run([
                proj_tool,
                "-z", str(drizzle),
                str(weight_map_weighted),
                str(weight_map_proj),
                str(skel_hdr)
            ], stdout=subprocess.PIPE if quiet else None, stderr=subprocess.PIPE)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to project weight map {weight_map_weighted} (exit {result.returncode}): {result.stderr.decode(errors='replace')}")

    # Clean up background-subtracted temp file (it has been projected into proj_dir)
    if bgsub_file is not None and Path(bgsub_file).exists():
        Path(bgsub_file).unlink()

    return {
        'image': str(output_proj),
        'weight_map': str(weight_map_proj) if has_weight_map else None
    }

def _convert_skeleton_if_needed(skel_path):
    """If the skeleton header has a projection mProjectPP can't handle (e.g. ZPN),
    convert it to TAN+SIP in-place so mProjectPX and mAdd use the same WCS."""
    from pyrt.cli.mProjectPX import (read_template_header,
                                      check_projection_handling,
                                      write_template_hdr)
    from pyrt.cli.zpn_to_tan import zpn_to_tan_mesh
    try:
        header = read_template_header(skel_path)
        if check_projection_handling(header) == 'convert':
            fitter, rms = zpn_to_tan_mesh(header, ngrid=200, sip_order=3)
            write_template_hdr(fitter, header['NAXIS1'], header['NAXIS2'], skel_path)
            print(f"Skeleton: converted {header.get('CTYPE1','?')[5:]} → TAN+SIP (RMS {rms:.3f} px)")
    except Exception as e:
        import traceback
        print(f"Warning: skeleton projection check/conversion failed: {e}")
        traceback.print_exc()


def combine_images_montage(output, inputs, weights, skeleton_file, args):
    """Combine images using Montage's mProjectPP/mProjectPX and mAdd.

    Handles parallel projection, weight map processing, and final combination.

    Args:
        output: Output FITS file path
        inputs: List of input FITS file paths
        weights: Dictionary mapping file paths to weights
        skeleton_file: Path to skeleton header file
        args: Parsed command-line arguments
    """
    home_tmp = Path.home() / "tmp"
    home_tmp.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=home_tmp) as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy skeleton header to temp directory, converting ZPN→TAN+SIP if needed.
        # mProjectPP handles ZPN input images internally, but mAdd also needs the
        # skeleton to match the projected images — so the conversion must happen here
        # once, and all downstream tools (mProjectPX, mAdd) use the same header.
        subprocess.run(
            ["cp", skeleton_file, str(tmpdir / "skel.hdr")],
            check=True,
            capture_output=True,
            text=True
        )
        _convert_skeleton_if_needed(str(tmpdir / "skel.hdr"))

        # Create directories
        weighted_dir = tmpdir / "weighted"
        proj_dir = tmpdir / "projected"
        projw_dir = tmpdir / "projw"
        weights_dir = tmpdir / "weights"
        weights_dir.mkdir()
        weighted_dir.mkdir()
        proj_dir.mkdir()
        projw_dir.mkdir()

        quiet = getattr(args, 'quiet', False)

        # Prepare input data for parallel processing
        process_inputs = [
            (
                f,
                weights[f],
                weighted_dir,
                proj_dir,
                tmpdir / "skel.hdr",
                args.drizzle,
                weights_dir,
                projw_dir,
                args.background_subtract,
                quiet,
            )
            for f in inputs
            if weights[f] > 0
        ]

        # Process images and weight maps in parallel
        max_workers = min(mp.cpu_count(), len(process_inputs))
        if not quiet:
            print(f"Processing images using {max_workers} processes...")

        with mp.Pool(processes=max_workers) as pool:
            if quiet:
                results = list(tqdm(
                    pool.imap(process_single_image, process_inputs),
                    total=len(process_inputs),
                    desc="Projecting images",
                    unit="img",
                ))
            else:
                results = pool.map(process_single_image, process_inputs)
        processed_files = [r for r in results if r is not None]

        if not processed_files:
            raise RuntimeError("No valid processed images produced")

        # Separate image and weight map files
        image_files = [p['image'] for p in processed_files]
        weight_map_files = [p['weight_map'] for p in processed_files if p['weight_map'] is not None]

        # Create image table for main images
        img_tbl = tmpdir / "images.tbl"
        result = subprocess.run(
            ["mImgtbl", str(proj_dir), str(img_tbl)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create image table: {result.stdout} {result.stderr}")

        # Combine main images
        print("Combining projected images...")
        result = subprocess.run([
            "mAdd",
            "-p", str(proj_dir),
            str(img_tbl),
            str(tmpdir / "skel.hdr"),
            output
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to combine images: {result.stdout} {result.stderr}")

        print("Images combined with normalized weights")
        try:
            with fits.open(output) as hdul:
                max_value = np.max(hdul[0].data)
                print(f"Final combined image max pixel value: {max_value:.1f}")
        except Exception as e:
            print(f"Warning: Could not read final image stats: {str(e)}")

        # If we have weight maps, combine them and apply to final image
        if weight_map_files:
            print("Processing weight maps...")
            weight_output = str(tmpdir / "combined_weight.fits")

            # Create table for weight maps
            wmap_tbl = tmpdir / "wmaps.tbl"
            result = subprocess.run(
                ["mImgtbl", str(projw_dir), str(wmap_tbl)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create weight map table: {result.stdout} {result.stderr}")

            # Combine weight maps
            result = subprocess.run([
                "mAdd",
                "-p", str(projw_dir),
                str(wmap_tbl),
                str(tmpdir / "skel.hdr"),
                weight_output
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to combine weight maps: {result.stdout} {result.stderr}")

            # Apply combined weight map to final image
            print("Applying combined weight map...")
            try:
                with fits.open(output) as hdul, fits.open(weight_output) as whdul:
                    hdul[0].data /= whdul[0].data
                    hdul.writeto(output, overwrite=True)
            except Exception as e:
                raise RuntimeError(f"Failed to apply combined weight map: {str(e)}")

        # Save output weight map for future coaddition if requested
        if args.keep_weightmap:
            print("Computing output weight map...")
            with fits.open(output) as ref_hdul:
                ref_shape = ref_hdul[0].data.shape
                ref_header = ref_hdul[0].header.copy()
            wmap_sum = np.zeros(ref_shape, dtype=np.float32)
            for proc_input, proc_result in zip(process_inputs, results):
                if proc_result is None:
                    continue
                w_i = proc_input[1]  # scalar weight
                if proc_result['weight_map'] is not None:
                    # Projected weight map already contains w_i (fixed above)
                    with fits.open(proc_result['weight_map']) as whdul:
                        wmap_sum += np.nan_to_num(whdul[0].data, nan=0.0).astype(np.float32)
                else:
                    # No per-pixel weight map: assume uniform coverage, weight by scalar w_i
                    with fits.open(proc_result['image']) as ihdul:
                        coverage = np.isfinite(ihdul[0].data) & (ihdul[0].data != 0)
                        wmap_sum += w_i * coverage.astype(np.float32)
            fits.writeto(args.keep_weightmap, wmap_sum, header=ref_header)
            print(f"Output weight map saved to {args.keep_weightmap}")

    update_output_header(output, inputs, weights, args)

    if args.keep_weightmap:
        result = subprocess.run(
            ["fitsheader", "-w", f"WGHTFILE={args.keep_weightmap}", output],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Warning: Could not set WGHTFILE header in {output}: {result.stderr}")

def combine_images_direct(output, inputs, skeleton_file, args):
    """Stack pre-projected images directly via mAdd, skipping reprojection.

    Frames must already be on the same WCS grid as the skeleton.
    Reads WGHT scalar weight from each frame's FITS header (default: 1.0).
    Per-pixel weights from WGHTFILE headers are used when present.

    Always writes a weight map alongside the output image.

    Args:
        output: Output FITS file path
        inputs: List of input FITS file paths (pre-projected)
        skeleton_file: Path to skeleton header file for mAdd
        args: Parsed command-line arguments
    """
    # Determine output weight map path
    stem = os.path.splitext(output)[0]
    wmap_output = args.keep_weightmap if args.keep_weightmap else stem + 'w.fits'

    # Read scalar weights from headers
    weights = {}
    for f in inputs:
        with fits.open(f) as hdul:
            w = float(hdul[0].header.get('WGHT', 1.0))
            if 'WGHT' not in hdul[0].header:
                print(f"Warning: {f} has no WGHT keyword, using 1.0")
            weights[f] = w

    total_w = sum(weights.values())
    n = len(inputs)
    # Normalize so weights sum to N: mAdd averages (divides by N), so this
    # recovers a proper weighted average in the final result.
    norm_weights = {f: w * n / total_w for f, w in weights.items()}

    print(f"Direct stacking {n} pre-projected frames (weight sum={total_w:.4g})...")
    for f, w in weights.items():
        print(f"  {os.path.basename(f)}: WGHT={w:.6g}  (normalized={norm_weights[f]:.6g})")

    home_tmp = Path.home() / "tmp"
    home_tmp.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=home_tmp) as tmpdir:
        tmpdir = Path(tmpdir)

        skel = tmpdir / "skel.hdr"
        subprocess.run(["cp", skeleton_file, str(skel)], check=True, capture_output=True)

        staged_dir = tmpdir / "staged"
        stagew_dir = tmpdir / "stagew"
        staged_dir.mkdir()
        stagew_dir.mkdir()

        has_any_wmap = False

        for f in inputs:
            base = Path(f).stem
            w = norm_weights[f]

            with fits.open(f) as hdul:
                header = hdul[0].header
                data = hdul[0].data
                wghtfile = header.get('WGHTFILE')
                has_wmap = bool(wghtfile and os.path.exists(wghtfile))

                if has_wmap:
                    with fits.open(wghtfile) as whdul:
                        wmap = whdul[0].data.astype(np.float64)
                        wmap_max = np.max(wmap)
                        if wmap_max > 0:
                            wmap /= wmap_max
                        weighted_data = (data.astype(np.float64) * w * wmap).astype(np.float32)
                        scaled_wmap = (wmap * w).astype(np.float32)
                        wmap_stem = Path(wghtfile).stem
                        fits.writeto(stagew_dir / f"{wmap_stem}.fits", scaled_wmap,
                                     header=whdul[0].header)
                        # Copy area file for the weight map if present
                        wmap_area = Path(wghtfile).parent / f"{wmap_stem}_area.fits"
                        if wmap_area.exists():
                            subprocess.run(["cp", str(wmap_area),
                                            str(stagew_dir / f"{wmap_stem}_area.fits")],
                                           check=True, capture_output=True)
                    has_any_wmap = True
                else:
                    weighted_data = (data.astype(np.float64) * w).astype(np.float32)

                fits.writeto(staged_dir / f"{base}.fits", weighted_data, header=header)
                # Copy the companion area file so mAdd can find it
                area_src = Path(f).parent / f"{base}_area.fits"
                if area_src.exists():
                    subprocess.run(["cp", str(area_src), str(staged_dir / f"{base}_area.fits")],
                                   check=True, capture_output=True)
                else:
                    print(f"Warning: area file not found for {f}: {area_src}")

        # mAdd on staged images
        img_tbl = tmpdir / "images.tbl"
        result = subprocess.run(["mImgtbl", str(staged_dir), str(img_tbl)],
                                 capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mImgtbl failed: {result.stdout} {result.stderr}")

        print("Combining staged frames with mAdd...")
        result = subprocess.run(
            ["mAdd", "-p", str(staged_dir), str(img_tbl), str(skel), output],
            capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.returncode != 0:
            raise RuntimeError(f"mAdd failed (rc={result.returncode}): {result.stdout} {result.stderr}")

        if has_any_wmap:
            wmap_tbl = tmpdir / "wmaps.tbl"
            result = subprocess.run(["mImgtbl", str(stagew_dir), str(wmap_tbl)],
                                     capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"mImgtbl (wmaps) failed: {result.stdout} {result.stderr}")

            wmap_tmp = str(tmpdir / "combinedw_tmp.fits")
            print("Combining weight maps with mAdd...")
            result = subprocess.run(
                ["mAdd", "-p", str(stagew_dir), str(wmap_tbl), str(skel), wmap_tmp],
                capture_output=True, text=True
            )
            if result.stdout:
                print(result.stdout.strip())
            if result.returncode != 0:
                raise RuntimeError(f"mAdd (wmaps) failed (rc={result.returncode}): {result.stdout} {result.stderr}")

            # Divide by combined weight map to normalise the weighted average
            with fits.open(output, mode='update') as hdul, fits.open(wmap_tmp) as whdul:
                wdata = whdul[0].data
                hdul[0].data = np.where(wdata > 0,
                                        hdul[0].data / wdata, np.nan).astype(np.float32)
                hdul.flush()

            subprocess.run(["cp", wmap_tmp, wmap_output], check=True, capture_output=True)
        else:
            # No per-pixel weight maps: derive coverage weight map from combined image
            with fits.open(output) as hdul:
                coverage = np.isfinite(hdul[0].data) & (hdul[0].data != 0)
                wmap_hdr = hdul[0].header.copy()
            fits.writeto(wmap_output, (coverage.astype(np.float32) * total_w / n), header=wmap_hdr)

    # Tag the output with WGHTFILE
    result = subprocess.run(
        ["fitsheader", "-w", f"WGHTFILE={wmap_output}", output],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Warning: Could not set WGHTFILE in {output}: {result.stderr}")

    update_output_header(output, inputs, norm_weights, args)

    print(f"Combined image written to {output}")
    print(f"Weight map written to {wmap_output}")


# ============================================================================
# Argument Parsing and Main
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Combine FITS images with intelligent weighting and outlier rejection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic mode - select images, calculate optimal frame, combine
  %(prog)s input*.fits

  # Specify output file
  %(prog)s -o output.fits input*.fits

  # Uniform weighting (no photometric calibration required)
  %(prog)s -u input*.fits

  # Calculate optimal WCS skeleton only (no combining)
  %(prog)s --skeleton-only skel.hdr input*.fits

  # Manual mode - use provided skeleton, no outlier rejection
  %(prog)s --skeleton custom.hdr --no-selection input*.fits

  # GRB mode with time-variable source
  %(prog)s --grb 2459000.5,17.0,-1.2 grb*.fits

  # Use measured magnitudes from photometry file
  %(prog)s --photfile dophot.dat transient*.fits

  # Custom frame calculation parameters
  %(prog)s --sampling-factor 16 --scale-factor 2.0 *.fits
        """
    )

    # Positional arguments
    parser.add_argument("inputs", nargs="+", help="Input FITS files to combine")

    # Pre-projected mode
    parser.add_argument("-P", "--pre-projected", action="store_true",
                       help="Stack pre-projected frames directly via mAdd. Skips selection, "
                            "frame calculation, and reprojection. Reads WGHT scalar weight "
                            "from each frame's header; per-pixel weights from WGHTFILE. "
                            "Always writes a weight map (combinedw.fits or -W path). "
                            "Use --skel to supply the target frame (required by mAdd).")

    # Output control
    parser.add_argument("-o", "--output", default="combined.fits",
                       help="Output FITS file (default: combined.fits)")
    parser.add_argument("--skeleton-only", metavar="FILE", nargs='?', const='skeleton.hdr',
                       help="Calculate optimal WCS, write skeleton to FILE, and exit (no combining). Default: skeleton.hdr")

    # Selection control
    parser.add_argument("--no-selection", action="store_true",
                       help="Skip outlier rejection, use all input files")

    # Frame calculation control
    parser.add_argument("--skeleton", "-S", "--skel", metavar="FILE",
                       help="Use provided skeleton header (skip frame calculation)")
    parser.add_argument("--sampling-factor", type=int, default=8,
                       help="Sampling factor for WCS calculations (default: 8)")
    parser.add_argument("--scale-factor", type=float, default=1.5,
                       help="Factor to scale field size (default: 1.5)")
    parser.add_argument("--zoom-factor", type=float, default=1.0,
                       help="Additional zoom factor for output (default: 1.0)")
    parser.add_argument("--out", action="store_true",
                       help="Use partial coverage mode: largest rectangle with any coverage (default: full coverage by all images)")

    # Weighting and photometry
    parser.add_argument("-u", "--uniform", action="store_true",
                       help="Use uniform weighting (equal weight for all images)")
    parser.add_argument("-g", "--gain", type=float, default=2.3,
                       help="CCD gain in e-/ADU (default: 2.3)")
    parser.add_argument("-b", "--brightness", type=float, default=17.0,
                       help="Reference source brightness in magnitudes (default: 17.0)")
    parser.add_argument("--grb", metavar="T0,MAG,DECAY",
                       help="GRB parameters: T0 (JD), magnitude at 1d, decay rate (mags/decade)")
    parser.add_argument("--photfile", metavar="FILE",
                       help="Photometry file with measured magnitudes (dophot output format: col 1=filename, col 12=mag, col 13=magerr)")

    # Processing options
    parser.add_argument("-B", "--background-subtract", action="store_true",
                       help="Subtract background from each image before combining (uses pyrt-phcat -B)")
    parser.add_argument("-W", "--keep-weightmap", metavar="FILE", nargs='?', const='__auto__',
                       help="Save output weight map for future coaddition (default name: {output}w.fits). "
                            "The weight map is the scalar-weighted sum of per-pixel weights across all frames.")
    parser.add_argument("-z", "--drizzle", type=float, default=1.0,
                       help="Drizzle scale factor, range (0, 2] (default: 1.0)")
    parser.add_argument("--num-processes", type=int, default=None,
                       help="Number of parallel processes (default: auto)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show full output from mProject and weight computation (default: progress bars)")

    args = parser.parse_args()
    args.quiet = not args.verbose

    # Parse GRB parameters if provided
    if args.grb:
        try:
            parts = args.grb.split(',')
            if len(parts) != 3:
                parser.error("GRB parameters must be exactly 3 values: T0,mag,decay")
            args.t0, args.brightness, args.rate = map(float, parts)
        except ValueError as e:
            parser.error(f"Error parsing GRB parameters: {e}")
    else:
        args.t0 = None
        args.rate = None

    # Validation
    if args.t0 is not None and args.rate is None:
        parser.error("GRB decay rate required when T0 is specified")
    if args.rate is not None and args.t0 is None:
        parser.error("GRB T0 required when decay rate is specified")

    if args.drizzle <= 0 or args.drizzle > 2:
        parser.error("Drizzle factor must be between 0 and 2")

    if args.gain <= 0:
        parser.error("Gain must be positive")

    # Resolve auto weight map filename
    if args.keep_weightmap == '__auto__':
        stem = os.path.splitext(args.output)[0]
        args.keep_weightmap = stem + 'w.fits'

    # Critical safety check: Prevent overwriting existing FITS files
    if args.skeleton_only:
        if os.path.exists(args.skeleton_only):
            parser.error(f"Operation would overwrite an existing file: {args.skeleton_only}")
    elif os.path.exists(args.output):
        parser.error(f"Output file {args.output} already exists")

    if args.keep_weightmap and os.path.exists(args.keep_weightmap):
        parser.error(f"Weight map output file already exists: {args.keep_weightmap}")

    # In pre-projected mode the weight map is always written; check the auto path too
    if args.pre_projected and not args.keep_weightmap:
        auto_wmap = os.path.splitext(args.output)[0] + 'w.fits'
        if os.path.exists(auto_wmap):
            parser.error(f"Weight map output file already exists: {auto_wmap}")

    return args

def main():
    """Main execution function."""
    args = parse_arguments()

    print("=" * 70)
    print("combine-images: Intelligent FITS Image Combination")
    print("=" * 70)

    # Pre-projected fast path: skip selection, frame calc, and weight computation
    if args.pre_projected:
        print("\nMode: Pre-projected (direct mAdd stack)")
        skeleton_file = resolve_skeleton(args.skeleton)
        if skeleton_file:
            print(f"Using skeleton: {skeleton_file}")
            if not os.path.exists(skeleton_file):
                raise FileNotFoundError(f"Skeleton file not found: {skeleton_file}")
        else:
            # Derive skeleton from the first frame's header
            print("No skeleton provided — deriving from first frame header")
            first = args.inputs[0]
            with fits.open(first) as hdul:
                derived_hdr = hdul[0].header
            skeleton_file = 'skel_derived.hdr'
            write_montage_header(dict(derived_hdr), skeleton_file)
            print(f"Derived skeleton written to {skeleton_file}")

        combine_images_direct(args.output, args.inputs, skeleton_file, args)

        print("\n" + "=" * 70)
        print(f"Successfully stacked {len(args.inputs)} pre-projected frames into {args.output}")
        print("=" * 70)
        return

    # Stage 1: Image Selection
    if args.no_selection:
        print("\nStage 1: Image Selection - SKIPPED (using all input files)")
        valid_inputs = args.inputs
    else:
        print("\nStage 1: Image Selection - Performing spatial clustering")
        valid_inputs, rejected = find_valid_images_cluster(args.inputs)
        if rejected:
            print(f"Rejected {len(rejected)} outlier(s)")

    if not valid_inputs:
        raise ValueError("No valid input files remaining after selection")

    print(f"Selected {len(valid_inputs)} images for combination")

    # Stage 2: WCS Frame Calculation
    if args.skeleton_only:
        # Skeleton-only mode: calculate frame and exit
        print(f"\nStage 2: WCS Frame - Calculating optimal frame (skeleton-only mode)")
        optimal_header = find_optimal_frame(
            valid_inputs,
            sampling_factor=args.sampling_factor,
            scale_factor=args.scale_factor,
            zoom_factor=args.zoom_factor,
            num_processes=args.num_processes,
            mode='out' if args.out else 'in'
        )
        write_montage_header(optimal_header, args.skeleton_only)
        print("\n" + "=" * 70)
        print(f"Skeleton header written to {args.skeleton_only}")
        print("=" * 70)
        return
    elif args.skeleton:
        print(f"\nStage 2: WCS Frame - SKIPPED (using provided skeleton: {args.skeleton})")
        skeleton_file = resolve_skeleton(args.skeleton)
        if not os.path.exists(skeleton_file):
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_file}")
    else:
        print("\nStage 2: WCS Frame - Calculating optimal frame" +
              (" (partial coverage)" if args.out else ""))
        optimal_header = find_optimal_frame(
            valid_inputs,
            sampling_factor=args.sampling_factor,
            scale_factor=args.scale_factor,
            zoom_factor=args.zoom_factor,
            num_processes=args.num_processes,
            mode='out' if args.out else 'in'
        )
        skeleton_file = 'skel.hdr'
        write_montage_header(optimal_header, skeleton_file)

    # Stage 3: Weight Calculation
    photometry = None
    if args.uniform:
        print("\nStage 3: Weight Calculation - Using uniform weights")
    else:
        print("\nStage 3: Weight Calculation - Computing optimal weights")
        if args.photfile:
            print(f"Using measured magnitudes from: {args.photfile}")
            photometry = parse_photometry_file(args.photfile, valid_inputs)
            print(f"Loaded magnitudes for {len(photometry)} files")
        elif args.t0 is not None:
            print(f"GRB mode: T0={args.t0}, mag@1d={args.brightness}, decay={args.rate} mag/decade")

    weights, skipped = compute_weights(
        valid_inputs,
        args.gain,
        args.t0,
        args.brightness,
        args.rate,
        uniform=args.uniform,
        photometry=photometry,
        quiet=args.quiet,
    )

    # Report skipped files
    if skipped:
        print(f"\nWarning: {len(skipped)} file(s) excluded from combination (missing MAGZERO):")
        for f in skipped:
            print(f"  - {f}")
        print("\nNote: To include all files with uniform weighting, use the --uniform (-u) option")

        # Check if ALL files were skipped
        valid_count = sum(1 for w in weights.values() if w > 0)
        if valid_count == 0:
            raise ValueError(
                "All files were excluded due to missing MAGZERO.\n"
                "Use --uniform (-u) to combine with equal weighting instead."
            )

    # Stage 4: Image Combination
    print("\nStage 4: Image Combination - Reprojecting and combining")
    if os.path.exists(args.output):
        raise ValueError(f"Output file {args.output} already exists")

    combine_images_montage(args.output, valid_inputs, weights, skeleton_file, args)

    # Final summary
    valid_count = sum(1 for w in weights.values() if w > 0)
    print("\n" + "=" * 70)
    print(f"Successfully combined {valid_count} images into {args.output}")
    print("=" * 70)

if __name__ == "__main__":
    main()
