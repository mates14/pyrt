#!/usr/bin/env python3

"""
Filter determination and validation utilities for photometric calibration.

This module provides unified filter handling for dophot3, including:
- Heuristic filter matching based on wavelength
- Statistical filter discovery using photometric fitting
- Filter validation with configurable strictness levels
- Cross-catalog filter compatibility checking

See filter-estimation.md for detailed documentation of the approach.
"""

import logging
import sys
import numpy as np

import fotfit
from catalog import Catalog, CatalogFilter


def get_catalog_filters(catalog_name):
    """
    Get filter information for a catalog without instantiating it.

    Args:
        catalog_name: Name of the catalog (e.g., 'atlas@localhost', 'makak', etc.)

    Returns:
        dict: Dictionary of CatalogFilter objects
    """
    catalog_info = Catalog.KNOWN_CATALOGS.get(catalog_name)
    if catalog_info is None:
        raise ValueError(f"Unknown catalog: {catalog_name}")
    return catalog_info['filters']


def find_closest_filter_by_wavelength(target_wavelength, available_filters):
    """Find the filter closest in wavelength to the target"""
    if not available_filters:
        return None

    closest_filter = None
    min_diff = float('inf')

    for cat_filter in list(available_filters.values()):
        if not isinstance(cat_filter, CatalogFilter):
            logging.warning(f"Skipping invalid filter object: {cat_filter}")
            continue
        wavelength = cat_filter.effective_wl
        if wavelength is not None:
            diff = abs(wavelength - target_wavelength)
            if diff < min_diff:
                min_diff = diff
                closest_filter = cat_filter

    return closest_filter


def find_compatible_schema(available_filters, filter_schemas, basemag):
    """Find compatible schema that includes the base filter"""
    available_filter_set = set(available_filters.keys())

    for schema_name, schema_filters in filter_schemas.items():
        if basemag not in schema_filters:
            continue
        if set(schema_filters).issubset(available_filter_set):
            logging.info(f"Photometric schema: {schema_name} with base filter {basemag}")
            return schema_name

    logging.warning(f"No compatible filter schema found that includes base filter {basemag}")
    return None


def get_base_filter(det, options, catalog_name):
    '''
    Set up or guess the base filter to use for fitting, selecting the closest available filter
    by wavelength when necessary, and determine the appropriate filter schema.

    Args:
        det: Detection table with metadata
        options: Command line options
        catalog_name: Name of the catalog to use

    Returns:
        tuple: (base_filter_name, photometric_system, schema_name)
    '''
    
    # Get available filters for this catalog
    available_filters = get_catalog_filters(catalog_name)
    if not available_filters:
        logging.warning(f"Catalog does not contain filters. Using Sloan_r + AB")
        return 'Sloan_r', 'AB', None

    # Map current filter name to catalog filter
    filter_name = det.meta.get('FILTER', 'Sloan_r')

    if filter_name in available_filters:
        basemag = available_filters[filter_name]
    else:
        # Find closest matching filter by wavelength
        FILTER_WAVELENGTHS = {
            'U': 3600, 'B': 4353, 'V': 5477, 'R': 6349, 'I': 8797,
            'g': 4810, 'r': 6170, 'i': 7520, 'z': 8660, 'y': 9620,
            'G': 5890, 'BP': 5050, 'RP': 7730,
            'Sloan_g': 4810, 'Sloan_r': 6170, 'Sloan_i': 7520, 'Sloan_z': 8660,
            'Johnson_U': 3600, 'Johnson_B': 4353, 'Johnson_V': 5477,
            'Johnson_R': 6349, 'Johnson_I': 8797, 'N': 6000
        }

        target_wavelength = FILTER_WAVELENGTHS.get(filter_name)
        logging.info(f"Filter {filter_name} has wavelength {target_wavelength}")
        if target_wavelength:
            closest = find_closest_filter_by_wavelength(target_wavelength, available_filters)
            if closest:
                basemag = closest

    # Find compatible schema that includes basemag
    schema_name = find_compatible_schema(available_filters, options.filter_schemas, basemag.name)

    # Determine photometric system
    fit_in_johnson = basemag.system

    return basemag.name, fit_in_johnson, schema_name


def select_best_filter(data, metadata):
    """
    Fast filter discovery using direct correlation against catalog filters.
    Much faster than full fitting - correlates observed vs catalog magnitudes.

    Args:
    data (PhotometryData): Object containing all photometry data.
    metadata (list): List of metadata for each image.

    Returns:
    str: The name of the best filter to use for initial calibration.
    """
    available_filters = data.get_filter_columns()
    best_filter = None
    best_correlation = -1
    
    # Get observed magnitudes (instrumental magnitudes from detection)
    try:
        obs_mags = data.get_arrays('y')[0]  # y = catalog magnitude in current filter
    except Exception as e:
        raise ValueError(f"Cannot get observed magnitudes for correlation: {e}")

    print(f"Testing correlation with {len(available_filters)} available filters...")
    
    for filter_name in available_filters:
        try:
            # Temporarily set filter to get catalog magnitudes
            original_filter = data._current_filter
            data.set_current_filter(filter_name)
            cat_mags = data.get_arrays('y')[0]  # Catalog mags in this filter
            
            # Compute correlation coefficient (Pearson)
            # Handle potential NaN values by using only valid pairs
            valid_mask = np.isfinite(obs_mags) & np.isfinite(cat_mags)
            if np.sum(valid_mask) < 10:  # Need minimum points for reliable correlation
                correlation = -1
            else:
                correlation = np.corrcoef(obs_mags[valid_mask], cat_mags[valid_mask])[0, 1]
                if np.isnan(correlation):
                    correlation = -1
            
            print(f"Filter {filter_name}: correlation = {correlation:.3f} (n={np.sum(valid_mask)} stars)")
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_filter = filter_name
                
        except Exception as e:
            print(f"Error testing filter {filter_name}: {str(e)}")
            correlation = -1
        finally:
            # Restore original filter
            if 'original_filter' in locals() and original_filter:
                data.set_current_filter(original_filter)

    if best_filter is None or best_correlation < 0.5:
        if best_filter is None:
            raise ValueError("No valid filter correlations found. Check the input data.")
        else:
            print(f"WARNING: Best correlation is only {best_correlation:.3f}, filter identification may be unreliable")

    print(f"Best filter: {best_filter} with correlation: {best_correlation:.3f}")
    return best_filter


def determine_filter(det, options, catalog_name):
    """
    Phase 1: Heuristic filter determination for initial catalog loading.
    Always runs early in pipeline, regardless of filter_check mode.
    
    Args:
        det: Detection table with FITS metadata
        options: Command line options
        catalog_name: Catalog to use for filter matching
    
    Returns:
        tuple: (filter_name, photometric_system, schema_name)
        
    Side effects:
        - Updates det.meta['PHFILTER'], det.meta['PHSYSTEM'], det.meta['PHSCHEMA']
    """
    # Get initial filter assignment from header/heuristic matching
    header_filter, photometric_system, schema_name = get_base_filter(det, options, catalog_name)
    
    # Update metadata
    det.meta['PHFILTER'] = header_filter
    det.meta['PHSYSTEM'] = photometric_system  
    det.meta['PHSCHEMA'] = schema_name
    
    logging.info(f"Heuristic filter selection: {header_filter} ({photometric_system} system)")
    
    return header_filter, photometric_system, schema_name


