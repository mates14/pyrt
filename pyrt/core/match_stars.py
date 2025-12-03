import os
import sys
import time
import logging
import argparse
from contextlib import suppress
import subprocess

import numpy as np

import astropy.wcs

import concurrent.futures
from copy import deepcopy

# This is to silence a particular annoying warning (MJD not present in a fits file)
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

from sklearn.neighbors import KDTree

from pyrt.core import zpnfit
from pyrt.core import fotfit
#from catalogs import get_atlas, get_catalog
from pyrt.catalog.catalog import Catalog, CatalogFilter
from pyrt.cli.cat2det import remove_junk
from pyrt.core.refit_astrometry import refit_astrometry
from pyrt.utils.file_utils import try_det, try_img, write_region_file
from pyrt.utils.dophot_config import parse_arguments
#from config import load_config
from pyrt.core.data_handling import PhotometryData, make_pairs_to_fit, compute_initial_zeropoints

def match_stars(det, cat, imgwcs, idlimit=2.0):
    """
    Match detected stars with catalog stars using KDTree.

    Args:
        det: Detection table with X_IMAGE, Y_IMAGE
        cat: Catalog table with radeg, decdeg
        imgwcs: WCS object for coordinate transformation
        idlimit: Match radius in pixels (default: 2.0)

    Returns:
        nearest_indices: Indices into original catalog
    """
    # Transform catalog coordinates to pixel space
    try:
        cat_x, cat_y = imgwcs.all_world2pix(cat['radeg'], cat['decdeg'], 1)
    except astropy.wcs.wcs.NoConvergence as e:
        # Some catalog coordinates failed to converge (likely outside valid WCS region)
        # This is normal - catalog queries include stars around the field, some outside image
        logging.warning(f"WCS transformation: {len(e.divergent) if e.divergent is not None else 0} catalog stars failed to converge")

        # Get best solution and filter out divergent points
        if e.best_solution is None or len(e.best_solution) == 0:
            logging.error("NoConvergence with no best_solution - WCS completely broken")
            return None

        # best_solution is shape (2, N) for x, y arrays
        cat_x = e.best_solution[0]
        cat_y = e.best_solution[1]

        # Create mask for valid (non-NaN, finite) coordinates
        valid_mask = np.isfinite(cat_x) & np.isfinite(cat_y)

        if not np.any(valid_mask):
            logging.error("All catalog stars failed WCS transformation")
            return None

        # Filter catalog to only valid stars
        n_removed = len(cat) - np.sum(valid_mask)
        cat = cat[valid_mask]
        cat_x = cat_x[valid_mask]
        cat_y = cat_y[valid_mask]

        logging.info(f"Filtered out {n_removed} catalog stars outside valid WCS region, continuing with {len(cat)} stars")

    except Exception as e:
        # Any other transformation failure is a real error
        logging.error(f"WCS transformation failed: {type(e).__name__}: {e}")
        return None

    # Create coordinate arrays
    det_coords = np.array([det['X_IMAGE'], det['Y_IMAGE']]).T
    cat_coords = np.array([cat_x, cat_y]).T

    # Build and query KDTree
    tree = KDTree(cat_coords)
    nearest_ind, _ = tree.query_radius(
        det_coords,
        r=idlimit,
        return_distance=True,
        count_only=False
    )

    return nearest_ind

def find_target(det, imgwcs, idlimit=2.0):
    """
    Find the target object in the detection list.

    Args:
        det: Detection table with target coordinates in meta
        imgwcs: WCS object for coordinate transformation
        idlimit: Match radius in pixels

    Returns:
        object: Matched detection or None if not found
    """
    try:
        if det.meta['OBJRA'] < -99 or det.meta['OBJDEC'] < -99:
            logging.info("Target was not defined")
            return None
    except:
        return None

    logging.info(f"Target coordinates: {det.meta['OBJRA']:.6f} {det.meta['OBJDEC']:+.6f}")

    try:
        # Transform target coordinates to pixel space
        target_x, target_y = imgwcs.all_world2pix(
            [det.meta['OBJRA']],
            [det.meta['OBJDEC']],
            1
        )

        if np.isnan(target_x[0]) or np.isnan(target_y[0]):
            logging.warning("Target transforms to NaN")
            return None

        # Create coordinate arrays
        target_coords = np.array([[target_x[0], target_y[0]]])
        det_coords = np.array([det['X_IMAGE'], det['Y_IMAGE']]).T

        # Build and query KDTree
        tree = KDTree(det_coords)
        object_ind, object_dist = tree.query_radius(
            target_coords,
            r=idlimit,
            return_distance=True,
            count_only=False
        )

        # Find closest match if any
        if len(object_ind[0]) > 0:
            mindist = np.inf
            target_match = None
            for idx, dist in zip(object_ind[0], object_dist[0]):
                if dist < mindist:
                    target_match = det[idx]
                    mindist = dist
            if target_match is not None:
                logging.info(f"Target is object id {target_match['NUMBER']} at distance {mindist:.2f} px")
                return target_match

    except Exception as e:
        logging.warning(f"Target search failed: {e}")

    return None

def estimate_rough_zeropoint(det, nearest_ind, cat, valid_cat_mask):
    """
    Estimate initial zeropoint from bright star matches.

    Args:
        det: Detection table
        matches: (indices, distances) from KDTree matching
        cat: Catalog table
        valid_cat_mask: Boolean mask for valid catalog entries

    Returns:
        float: Estimated zeropoint
    """

    # Get only entries with matches
    valid_matches = [i for i, inds in enumerate(nearest_ind) if len(inds) > 0]

    if not valid_matches:
        return 0.0

    # Get matched magnitudes
    inst_mags = det['MAG_AUTO'][valid_matches]
    cat_mags = []

    # Get catalog indices accounting for the valid_cat_mask
    for i, inds in enumerate(nearest_ind):
        if len(inds) > 0:
            cat_mags.append(cat[det.meta['PHFILTER']][inds[0]])

    cat_mags = np.array(cat_mags)

    # Use brightest 30% of stars for initial estimate
    bright_limit = np.percentile(inst_mags, 30)
    bright_mask = inst_mags < bright_limit

    if np.sum(bright_mask) > 0:
        zp = np.median(cat_mags[bright_mask] - inst_mags[bright_mask])
        logging.info(f"Initial zeropoint estimate: {zp:.3f} using {np.sum(bright_mask)} bright stars")
        return zp

    return 0.0

def get_catalog_with_dynamic_limit(det, estimated_zp, options):
    """
    Get catalog data with magnitude limit scaled from detection limit.

    Args:
        det: Detection table
        estimated_zp: Initial zeropoint estimate
        options: Command line options
        safety_margin: How many magnitudes above detection limit to use

    Returns:
        Catalog: New catalog instance with appropriate magnitude limit
    """

    # Calculate magnitude limit based on detection limit
    detection_limit = det.meta['LIMFLX3'] + estimated_zp
    maglim = detection_limit - options.margin

    # configuration option is a pure override
    if options.maglim is not None:
        maglim = options.maglim

    det.meta['MAGLIM'] = maglim

    logging.info(f"Detection limit: {detection_limit:.2f}, using catalog limit: {maglim:.2f}")

    # Get catalog with calculated magnitude limit
    enlarge = options.enlarge if options.enlarge is not None else 1.0
    catalog_name = 'makak' if options.makak else options.catalog

    cat = Catalog(
        ra=det.meta['CTRRA'],
        dec=det.meta['CTRDEC'],
        width=enlarge * det.meta['FIELD'],
        height=enlarge * det.meta['FIELD'],
        mlim=maglim,
        catalog=catalog_name
    )

    return cat

def estimate_magnitude_limit(field_size_deg, desired_stars, galactic_lat):
    """
    Estimate the magnitude limit needed to get desired number of stars
    using a simplified Bahcall-Soneira model.

    Args:
        field_size_deg (float): Field size in degrees
        desired_stars (int): Desired number of stars in the field
        galactic_lat (float): Absolute galactic latitude in degrees

    Returns:
        float: Estimated magnitude limit to get the desired number of stars
    """
    # Convert field size to square degrees
    area_sq_deg = field_size_deg * field_size_deg

    # Adjust constant based on galactic latitude
    abs_lat = abs(galactic_lat)
    if abs_lat < 20:
        # Near galactic plane - more stars
        C = 3.5
    elif abs_lat < 40:
        # Mid latitudes
        C = 3.8
    else:
        # High latitude fields
        C = 4.0

    # Using the formula N ≈ 10^(0.6m - C) * area
    # Solve for m given N:
    # log10(N/area) = 0.6m - C
    # m = (log10(N/area) + C) / 0.6
    mag_limit = (np.log10(desired_stars / area_sq_deg) + C) / 0.6

    return mag_limit

def process_image_with_dynamic_limits(det, options):
    """
    Process a single image with dynamic magnitude limits.

    Args:
        det: Detection table
        options: Command line options

    Returns:
        tuple: (catalog, matches, imgwcs, target) or (None, None, None, None) on failure
    """
    if True:
        # Set up WCS
        imgwcs = astropy.wcs.WCS(det.meta)

        target_match = find_target(det, imgwcs,
                                 idlimit=options.idlimit if options.idlimit else 2.0)

        enlarge = options.enlarge if options.enlarge is not None else 1.0

        field_size = enlarge * det.meta['FIELD']

        # If we have galactic coordinates, use them
        galactic_lat = abs(det.meta.get('GLAT', 45.0))  # default to mid-latitude if not available

        # Get star count estimate and recommended magnitude limit
        recommended_mag = estimate_magnitude_limit(field_size, 2000, galactic_lat)

        logging.info(f"Field size: {field_size:.2f} deg, using initial magnitude limit: {options.maglim or recommended_mag:.1f}")

        # Initial catalog search with bright stars
        cat = Catalog(
            ra=det.meta['CTRRA'],
            dec=det.meta['CTRDEC'],
            width=enlarge*det.meta['FIELD'],
            height=enlarge*det.meta['FIELD'],
            # Use bright stars for initial estimate
            mlim=options.maglim or recommended_mag,
            catalog='makak' if options.makak else options.catalog
        )

        # Check if catalog is empty (e.g., SDSS outside coverage area)
        if len(cat) == 0:
            logging.warning(f"Empty catalog from {options.catalog} - no reference stars available")
            return None, None, None, None

        if options.maglim is None:
            # this is for the adaptive magnitude limit
            # Match stars
            matches = match_stars(det, cat, imgwcs,
                                idlimit=options.idlimit if options.idlimit else 2.0)
            if matches is None:
                return None, None, None, None

            # Estimate zeropoint
            estimated_zp = estimate_rough_zeropoint(det, matches, cat, None)

            # Calculate magnitude limit based on detection limit
            detection_limit = det.meta['LIMFLX3'] + estimated_zp
            maglim = detection_limit - options.margin

            # configuration option is a pure override
            if options.maglim is not None:
                maglim = options.maglim

            det.meta['MAGLIM'] = maglim

            logging.info(f"Detection limit: {detection_limit:.2f}, using catalog limit: {maglim:.2f}")

            # Get catalog with calculated magnitude limit
            catalog_name = 'makak' if options.makak else options.catalog

            cat = Catalog(
                ra=det.meta['CTRRA'],
                dec=det.meta['CTRDEC'],
                width=enlarge * det.meta['FIELD'],
                height=enlarge * det.meta['FIELD'],
                mlim=maglim,
                catalog=catalog_name
            )

            # Check if catalog is empty (e.g., SDSS outside coverage area)
            if len(cat) == 0:
                logging.warning(f"Empty catalog from {catalog_name} - no reference stars available")
                return None, None, None, None

        logging.info(f"Matching {len(det)} objects from file with {len(cat)} objects from the catalog")
        # Final matching with full catalog
        final_matches = match_stars(det, cat, imgwcs,
                                  idlimit=options.idlimit if options.idlimit else 2.0)

        return cat, final_matches, imgwcs, target_match
