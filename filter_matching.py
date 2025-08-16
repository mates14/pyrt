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


