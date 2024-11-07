import os
import sys
import time
import logging
import argparse
from contextlib import suppress
import subprocess

import numpy as np

import astropy
import astropy.io.fits
import astropy.wcs
import astropy.table
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation

import concurrent.futures
from copy import deepcopy

# This is to silence a particular annoying warning (MJD not present in a fits file)
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

from sklearn.neighbors import KDTree

import zpnfit
import fotfit
#from catalogs import get_atlas, get_catalog
from catalog import Catalog, CatalogFilter
from cat2det import remove_junk
from refit_astrometry import refit_astrometry
from file_utils import try_det, try_img, write_region_file
from config import parse_arguments
#from config import load_config
from data_handling import PhotometryData, make_pairs_to_fit, compute_initial_zeropoints

def match_stars(det, cat, imgwcs, idlimit=2.0):
    """
    Match detected stars with catalog stars using KDTree.
    
    Args:
        det: Detection table with X_IMAGE, Y_IMAGE
        cat: Catalog table with radeg, decdeg
        imgwcs: WCS object for coordinate transformation
        idlimit: Match radius in pixels (default: 2.0)
        
    Returns:
        tuple: (nearest_indices, nearest_distances, valid_catalog_mask)
            - nearest_indices: Array of indices for each detection's matches
            - nearest_distances: Array of distances for each match
            - valid_catalog_mask: Boolean mask for valid catalog entries
    """
    # Transform catalog coordinates to pixel space
    try:
        cat_x, cat_y = imgwcs.all_world2pix(cat['radeg'], cat['decdeg'], 1)
    except (ValueError, KeyError):
        logging.warning("Astrometry transformation failed")
        return None, None, None
        
    # Create mask for valid catalog entries (on image)
    valid_cat_mask = ~np.any([
        np.isnan(cat_x),
        np.isnan(cat_y),
        cat_x < 0,
        cat_y < 0,
        cat_x > det.meta['IMGAXIS1'],
        cat_y > det.meta['IMGAXIS2']
    ], axis=0)
    
    if not np.any(valid_cat_mask):
        logging.warning("No valid catalog stars on image")
        return None, None, valid_cat_mask
    
    # Create coordinate arrays
    det_coords = np.array([det['X_IMAGE'], det['Y_IMAGE']]).T
    cat_coords = np.array([cat_x[valid_cat_mask], cat_y[valid_cat_mask]]).T
    
    # Build and query KDTree
    tree = KDTree(cat_coords)
    nearest_ind, nearest_dist = tree.query_radius(
        det_coords, 
        r=idlimit,
        return_distance=True,
        count_only=False
    )
    
    return nearest_ind, nearest_dist, valid_cat_mask

def estimate_rough_zeropoint(det, matches, cat, valid_cat_mask):
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
    nearest_ind, nearest_dist, valid_cat_mask = matches
    
    # Get only entries with matches
    valid_matches = [i for i, inds in enumerate(nearest_ind) if len(inds) > 0]
    
    if not valid_matches:
        return 0.0
        
    # Get matched magnitudes
    inst_mags = det['MAG_AUTO'][valid_matches]
    cat_mags = []
    
    # Get catalog indices accounting for the valid_cat_mask
    valid_cat_indices = np.where(valid_cat_mask)[0]
    for i, inds in enumerate(nearest_ind):
        if len(inds) > 0:
            # Map back to original catalog index
            orig_idx = valid_cat_indices[inds[0]]
            cat_mags.append(cat[det.meta['PHFILTER']][orig_idx])
    
    cat_mags = np.array(cat_mags)
    
    # Use brightest 30% of stars for initial estimate
    bright_limit = np.percentile(inst_mags, 30)
    bright_mask = inst_mags < bright_limit
    
    if np.sum(bright_mask) > 0:
        zp = np.median(cat_mags[bright_mask] - inst_mags[bright_mask])
        logging.info(f"Initial zeropoint estimate: {zp:.3f} using {np.sum(bright_mask)} bright stars")
        return zp
    
    return 0.0

def get_catalog_with_dynamic_limit(det, estimated_zp, options, safety_margin=1.25):
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
    maglim = detection_limit - safety_margin
    
    logging.info(f"Detection limit: {detection_limit:.2f}, using catalog limit: {maglim:.2f}")
    
    # Get catalog with calculated magnitude limit
    enlarge = options.enlarge if options.enlarge is not None else 1.0
    catalog_name = 'makak' if options.makak else (options.catalog or 'atlas@localhost')
    
    cat = Catalog(
        ra=det.meta['CTRRA'],
        dec=det.meta['CTRDEC'],
        width=enlarge * det.meta['FIELD'],
        height=enlarge * det.meta['FIELD'],
        mlim=maglim,
        catalog=catalog_name
    )
    
    return cat

def process_image_with_dynamic_limits(det, options):
    """
    Process a single image with dynamic magnitude limits.
    
    Args:
        det: Detection table
        options: Command line options
        
    Returns:
        tuple: (catalog, matches, imgwcs) or (None, None, None) on failure
    """
#    try:
    if True:
        # Set up WCS
        imgwcs = astropy.wcs.WCS(det.meta)
        
        # Initial catalog search with bright stars
        initial_cat = Catalog(
            ra=det.meta['CTRRA'],
            dec=det.meta['CTRDEC'],
            width=det.meta['FIELD'],
            height=det.meta['FIELD'],
            mlim=16.0,  # Use bright stars for initial estimate
            catalog='makak' if options.makak else (options.catalog or 'atlas@localhost')
        )
        
        # Match stars
        matches = match_stars(det, initial_cat, imgwcs, 
                            idlimit=options.idlimit if options.idlimit else 2.0)
        if matches[0] is None:
            return None, None, None
            
        # Estimate zeropoint
        estimated_zp = estimate_rough_zeropoint(det, matches, initial_cat, matches[2])
        
        # Get catalog with appropriate magnitude limit
        cat = get_catalog_with_dynamic_limit(det, estimated_zp, options)
        
        # Final matching with full catalog
        final_matches = match_stars(det, cat, imgwcs,
                                  idlimit=options.idlimit if options.idlimit else 2.0)
        
        return cat, final_matches, imgwcs
        
#    except Exception as e:
#        logging.error(f"Error processing image: {e}")
#        return None, None, None
