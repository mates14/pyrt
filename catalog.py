#!/usr/bin/python3

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import warnings
import os
import sys
import astropy.table
import astropy.io.ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from contextlib import suppress

@dataclass
class CatalogFilter:
    """Information about a filter in a catalog"""
    name: str           # Original filter name in catalog
    effective_wl: float # Effective wavelength in Angstroms
    system: str        # Photometric system (e.g., 'AB', 'Vega')
    error_name: Optional[str] = None  # Name of error column if available

class CatalogFilters:
    """Filter definitions for different catalogs"""

    # Pan-STARRS DR2 filters
    PANSTARRS = {
        'g': CatalogFilter('gMeanPSFMag', 4810, 'AB', 'gMeanPSFMagErr'),
        'r': CatalogFilter('rMeanPSFMag', 6170, 'AB', 'rMeanPSFMagErr'),
        'i': CatalogFilter('iMeanPSFMag', 7520, 'AB', 'iMeanPSFMagErr'),
        'z': CatalogFilter('zMeanPSFMag', 8660, 'AB', 'zMeanPSFMagErr'),
        'y': CatalogFilter('yMeanPSFMag', 9620, 'AB', 'yMeanPSFMagErr'),
    }

    # Gaia DR3 filters
    GAIA = {
        'G': CatalogFilter('phot_g_mean_mag', 5890, 'Vega', 'phot_g_mean_mag_error'),
        'BP': CatalogFilter('phot_bp_mean_mag', 5050, 'Vega', 'phot_bp_mean_mag_error'),
        'RP': CatalogFilter('phot_rp_mean_mag', 7730, 'Vega', 'phot_rp_mean_mag_error'),
    }

    # ATLAS filters
    ATLAS = {
        'Sloan_g': CatalogFilter('Sloan_g', 4810, 'AB'),
        'Sloan_r': CatalogFilter('Sloan_r', 6170, 'AB'),
        'Sloan_i': CatalogFilter('Sloan_i', 7520, 'AB'),
        'Sloan_z': CatalogFilter('Sloan_z', 8660, 'AB'),
        'J': CatalogFilter('J', 12000, 'AB'),
        'Johnson_B': CatalogFilter('Johnson_B', 4353, 'Vega'),
        'Johnson_V': CatalogFilter('Johnson_V', 5477, 'Vega'),
        'Johnson_R': CatalogFilter('Johnson_R', 6349, 'Vega'),
        'Johnson_I': CatalogFilter('Johnson_I', 8797, 'Vega'),
    }


class CatalogManager:
    """Unified catalog management with string-based catalog identification"""

    # Define available catalogs with their properties
    KNOWN_CATALOGS = {
        'ATLAS': {
            'description': 'Local ATLAS catalog',
            'filters': CatalogFilters.ATLAS,
            'epoch': 2015.5,
            'local': True,
            'service': 'local',
            'mag_splits': [
                ('00_m_16', 16),
                ('16_m_17', 17),
                ('17_m_18', 18),
                ('18_m_19', 19),
                ('19_m_20', 20)
            ]
        },
        'PANSTARRS': {
            'description': 'Pan-STARRS Data Release 2',
            'filters': CatalogFilters.PANSTARRS,
            'catalog': 'Panstarrs', #?
            'table': 'mean', #?
            'release': 'dr2', #?
            'epoch': 2015.5,
            'local': False,
            'service': 'MAST',
            'column_mapping': {
                'raMean': 'radeg',
                'decMean': 'decdeg',
                'gMeanPSFMag': 'gMeanPSFMag',
                'gMeanPSFMagErr': 'gMeanPSFMagErr',
                'rMeanPSFMag': 'rMeanPSFMag',
                'rMeanPSFMagErr': 'rMeanPSFMagErr',
                'iMeanPSFMag': 'iMeanPSFMag',
                'iMeanPSFMagErr': 'iMeanPSFMagErr',
                'zMeanPSFMag': 'zMeanPSFMag',
                'zMeanPSFMagErr': 'zMeanPSFMagErr',
                'yMeanPSFMag': 'yMeanPSFMag',
                'yMeanPSFMagErr': 'yMeanPSFMagErr'
            }
        },
        'GAIA_DR3': {
            'description': 'Gaia Data Release 3',
            'filters': CatalogFilters.GAIA,
            'epoch': 2016.0,
            'local': False,
            'service': 'Gaia',
            'catalog': 'gaiadr3.gaia_source'
        },
        'ATLAS_REFCAT2': {
            'description': 'ATLAS Reference Catalog 2',
            'filters': CatalogFilters.ATLAS,
            'epoch': 2015.5,
            'local': False,
            'service': 'VizieR',
            'catalog': 'J/ApJ/867/105',  # Updated catalog reference
            'column_mapping': {
                'RA_ICRS': 'radeg',
                'DE_ICRS': 'decdeg',
                'gmag': 'Sloan_g',
                'rmag': 'Sloan_r',
                'imag': 'Sloan_i',
                'zmag': 'Sloan_z',
                'Jmag': 'J',
                'e_gmag': 'Sloan_g_err',
                'e_rmag': 'Sloan_r_err',
                'e_imag': 'Sloan_i_err',
                'e_zmag': 'Sloan_z_err',
                'e_Jmag': 'J_err',
                'pmRA': 'pmra',
                'pmDE': 'pmdec'
            }
        },
        'MAKAK': {
            'description': 'Pre-filtered wide-field catalog',
            'filters': CatalogFilters.ATLAS,  # Using ATLAS filter definitions
            'epoch': 2015.5,  # Default epoch, could be overridden from FITS metadata
            'local': True,
            'service': 'local',
            'filepath': '/home/mates/test/catalog.fits'  # Default path, could be configurable
        }

    }


    def __init__(self, atlas_dir="/home/mates/cat/atlas", prefer_local=True):
        """
        Initialize CatalogManager.

        Args:
            atlas_dir (str): Path to local ATLAS catalog
            prefer_local (bool): Whether to prefer local catalogs when available
        """
        self.atlas_dir = atlas_dir
        self.prefer_local = prefer_local

    def get_catalog(self, ra, dec, width=0.25, height=0.25, mlim=17.0,
                   catalog='ATLAS', timeout=60):
        """Get catalog data from specified source."""
        if catalog not in self.KNOWN_CATALOGS:
            raise ValueError(f"Unknown catalog: {catalog}. "
                           f"Available catalogs: {list(self.KNOWN_CATALOGS.keys())}")

        cat_info = self.KNOWN_CATALOGS[catalog]

        # Try local first if preferred and available
        if self.prefer_local and cat_info['local']:
            try:
                if catalog == 'ATLAS':
                    cat = self._get_atlas_local(ra, dec, width, height, mlim)
                    if cat is not None:
                        return cat
                if catalog == 'MAKAK':
                    cat = self._get_makak_local(ra, dec, width, height, mlim)
                    if cat is not None:
                        return cat
            except Exception as e:
                warnings.warn(f"Local catalog access failed: {e}")

        # Fall back to online access
        try:
            service = cat_info['service']
            if service == 'MAST':
                return self._get_panstarrs_mast(ra, dec, width, height, mlim, timeout)
            elif service == 'Gaia':
                return self._get_gaia(ra, dec, width, height, mlim, timeout)
            elif service == 'VizieR':
                return self._get_atlas_vizier(ra, dec, width, height, mlim, timeout)
            elif service == 'local':
                pass
            else:
                raise ValueError(f"Unknown service type: {service}")

        except Exception as e:
            warnings.warn(f"Catalog query failed for {catalog} ({cat_info['service']}): {e}")
            return None

    @classmethod
    def list_available_catalogs(cls):
        """List all available catalog sources with descriptions"""
        return {name: info['description']
                for name, info in cls.KNOWN_CATALOGS.items()}

    def _get_atlas_local(self, ra, dec, width, height, mlim):
        """Get data from local ATLAS catalog"""
        cat = None

        try:
            # Get data for each magnitude range
            for dirname, maglim in self.KNOWN_CATALOGS['ATLAS']['mag_splits']:
                if mlim <= maglim:
                    continue

                directory = os.path.join(self.atlas_dir, dirname)
                try:
                    new_data = self._get_atlas_split(ra, dec, width, height,
                                                   directory, mlim)
                    if new_data is None:
                        continue

                    if cat is None:
                        cat = new_data
                    else:
                        cat = astropy.table.vstack([cat, new_data])
                except Exception as e:
                    warnings.warn(f"Failed to get data from {directory}: {e}")

            if cat is not None and len(cat) > 0:
                # Add transformed magnitudes
                self._add_transformed_magnitudes(cat)

                # Add proper metadata
                cat.meta['catalog'] = 'ATLAS'
                cat.meta['astepoch'] = self.KNOWN_CATALOGS['ATLAS']['epoch']
                cat.meta['filters'] = list(self.KNOWN_CATALOGS['ATLAS']['filters'].keys())

            return cat

        except Exception as e:
            warnings.warn(f"Error in _get_atlas_local: {e}")
            return None

    def _get_atlas_split(self, ra, dec, width, height, directory, mlim):
        """Get data from one magnitude split of ATLAS catalog"""
        atlas_ecsv_tmp = f"atlas{os.getpid()}.ecsv"
        try:
            cmd = f'atlas {ra} {dec} -rect {width},{height} '\
                  f'-dir {directory} -mlim {mlim:.2f} -ecsv > {atlas_ecsv_tmp}'
            os.system(cmd)
            return astropy.io.ascii.read(atlas_ecsv_tmp, format='ecsv')
        finally:
            with suppress(FileNotFoundError):
                os.remove(atlas_ecsv_tmp)

    def _add_transformed_magnitudes(self, cat):
        """Add transformed Johnson magnitudes to catalog"""
        gr = cat['Sloan_g'] - cat['Sloan_r']
        ri = cat['Sloan_r'] - cat['Sloan_i']
        iz = cat['Sloan_i'] - cat['Sloan_z']

        # Photometric transformations (2024 calibration)
        cat['Johnson_B'] = cat['Sloan_r'] + 1.490989 * gr + \
                          0.125787 * gr * gr - 0.022359 * gr*gr*gr + 0.186304
        cat['Johnson_V'] = cat['Sloan_r'] + 0.510236 * gr - 0.0337082
        cat['Johnson_R'] = cat['Sloan_r'] - 0.197420 * ri - \
                          0.083113 * ri * ri - 0.179943
        cat['Johnson_I'] = cat['Sloan_r'] - 0.897087 * ri - \
                          0.575316 * iz - 0.423971

    def _get_panstarrs_mast(self, ra, dec, width, height, mlim, timeout):
        """Get PanSTARRS DR2 data using their specific API"""
        try:
            from astroquery.mast import Catalogs
            # Convert width/height to radius in degrees
            radius = np.sqrt(width**2 + height**2) / 2
            
            # Create coordinate object
            coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            
            # Query parameters - note: no columns restriction
            constraints = {
                'nDetections.gt': 4,  # At least 5 detections
                'rMeanPSFMag.lt': mlim,  # Magnitude limit
                'qualityFlag.lt': 128  # Basic quality cut
            }
            
            # Make the query without specifying columns
            ps1 = Catalogs.query_region(
                coords,
                catalog="Panstarrs",
                radius=radius * u.deg,
                data_release="dr2",
                table="mean",
                **constraints
            )
            
            if len(ps1) == 0:
                return None
            
            # Create output catalog
            cat = astropy.table.Table()
            
            # Initialize columns from mapping
            column_mapping = self.KNOWN_CATALOGS['PANSTARRS']['column_mapping']
            our_columns = set(column_mapping.values())  # Use set to remove any duplicates
            for col in our_columns:
                cat[col] = np.zeros(len(ps1), dtype=np.float64)
            
            # Add proper motion columns (not provided by PanSTARRS)
            cat['pmra'] = np.zeros(len(ps1), dtype=np.float64)
            cat['pmdec'] = np.zeros(len(ps1), dtype=np.float64)
            
            # Map the data
            for ps1_name, our_name in column_mapping.items():
                if ps1_name in ps1.columns:
                    cat[our_name] = ps1[ps1_name].astype(np.float64)
            
            # Add metadata
            cat.meta['catalog'] = 'PANSTARRS'
            cat.meta['astepoch'] = self.KNOWN_CATALOGS['PANSTARRS']['epoch']
            cat.meta['filters'] = list(self.KNOWN_CATALOGS['PANSTARRS']['filters'].keys())
            
            return cat
            
        except Exception as e:
            warnings.warn(f"PanSTARRS query failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_atlas_vizier(self, ra, dec, width, height, mlim, timeout):
        """Get ATLAS RefCat2 data from VizieR with updated column mapping"""
        try:
            from astroquery.vizier import Vizier
            # Configure Vizier with correct column names
            column_mapping = self.KNOWN_CATALOGS['ATLAS_REFCAT2']['column_mapping']
            vizier = Vizier(
                columns=list(column_mapping.keys()),
                column_filters={
                    "rmag": f"<{mlim}"  # Magnitude limit in r-band
                },
                row_limit=-1
            )

            # Create coordinate object
            coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

            # Query VizieR
            result = vizier.query_region(
                coords,
                width=width * u.deg,
                height=height * u.deg,
                catalog=self.KNOWN_CATALOGS['ATLAS_REFCAT2']['catalog']
            )

            if not result or len(result) == 0:
                return None

            atlas = result[0]

            # Create output catalog
            cat = astropy.table.Table()

            # Initialize all columns from the mapping with zeros
            our_columns = set(column_mapping.values())  # Use set to remove any duplicates
            for col in our_columns:
                cat[col] = np.zeros(len(atlas), dtype=np.float64)

            # Map columns according to our mapping
            for vizier_name, our_name in column_mapping.items():
                if vizier_name in atlas.columns:
                    # Convert proper motions from mas/yr to deg/yr if needed
                    if vizier_name in ['pmRA', 'pmDE']:
                        cat[our_name] = atlas[vizier_name] / (3.6e6)
                    else:
                        cat[our_name] = atlas[vizier_name]

            # Add computed Johnson magnitudes
            self._add_transformed_magnitudes(cat)

            # Add metadata
            cat.meta['catalog'] = 'ATLAS_REFCAT2'
            cat.meta['astepoch'] = self.KNOWN_CATALOGS['ATLAS_REFCAT2']['epoch']
            cat.meta['filters'] = list(self.KNOWN_CATALOGS['ATLAS_REFCAT2']['filters'].keys())

            return cat

        except Exception as e:
            warnings.warn(f"VizieR ATLAS query failed: {e}")
            return None

    def _get_gaia(self, ra, dec, width, height, mlim, timeout):
        """Get Gaia DR3 data"""
        try:
            from astroquery.gaia import Gaia
            # Construct ADQL query with better formatting
            query = f"""
            SELECT
                source_id,
                ra, dec,
                pmra, pmdec,
                phot_g_mean_mag, phot_g_mean_flux_over_error,
                phot_bp_mean_mag, phot_bp_mean_flux_over_error,
                phot_rp_mean_mag, phot_rp_mean_flux_over_error,
                parallax, parallax_over_error,
                ruwe,
                visibility_periods_used
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                BOX('ICRS', {ra}, {dec}, {width}, {height}))
                AND phot_g_mean_mag < {mlim}
                AND ruwe < 1.4
                AND visibility_periods_used >= 8
                AND phot_g_mean_flux_over_error > 50
                AND phot_bp_mean_flux_over_error > 20
                AND phot_rp_mean_flux_over_error > 20
            """

            # Configure Gaia service
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            Gaia.ROW_LIMIT = -1

            # Execute query
            job = Gaia.launch_job_async(query)
            gaia_cat = job.get_results()

            if len(gaia_cat) == 0:
                return None

            # Create output catalog
            cat = astropy.table.Table()

            # Position and proper motion
            cat['radeg'] = gaia_cat['ra']
            cat['decdeg'] = gaia_cat['dec']
            cat['pmra'] = gaia_cat['pmra'] / (3.6e6)  # mas/yr to deg/yr
            cat['pmdec'] = gaia_cat['pmdec'] / (3.6e6)  # mas/yr to deg/yr

            # Add Gaia magnitudes and errors
            filters = self.KNOWN_CATALOGS['GAIA_DR3']['filters']
            for filter_name, filter_info in filters.items():
                cat[filter_info.name] = gaia_cat[filter_info.name]
                if filter_info.error_name:
                    # Convert flux_over_error to magnitude error
                    flux_over_error = gaia_cat[filter_info.name.replace('mag', 'flux_over_error')]
                    cat[filter_info.error_name] = 2.5 / (flux_over_error * np.log(10))

            # Add metadata
            cat.meta['catalog'] = 'GAIA_DR3'
            cat.meta['astepoch'] = self.KNOWN_CATALOGS['GAIA_DR3']['epoch']
            cat.meta['filters'] = list(filters.keys())

            return cat

        except Exception as e:
            warnings.warn(f"Gaia query failed: {e}")
            return None

    def _get_makak_local(self, ra, dec, width, height, mlim):
        """Get data from pre-filtered MAKAK catalog"""
        try:
            # Read the pre-filtered catalog
            cat = astropy.table.Table.read(self.KNOWN_CATALOGS['MAKAK']['filepath'])
            
            # Filter by field of view
            ctr = SkyCoord(ra*u.deg, dec*u.deg, frame='fk5')
            cat_coords = SkyCoord(cat['radeg']*u.deg, cat['decdeg']*u.deg, frame='fk5')
            within_field = cat_coords.separation(ctr) < (height*u.deg / 2)  # Using height as diameter
            cat = cat[within_field]
            
            if len(cat) > 0:
                # Set metadata
                cat.meta['catalog'] = 'MAKAK'
                cat.meta['astepoch'] = cat.meta.get('astepoch', 
                    self.KNOWN_CATALOGS['MAKAK']['epoch'])
                cat.meta['filters'] = list(self.KNOWN_CATALOGS['MAKAK']['filters'].keys())
                
                return cat
                
            return None
            
        except Exception as e:
            warnings.warn(f"MAKAK catalog access failed: {e}")
            return None

    @classmethod
    def set_makak_path(cls, filepath):
        """Update the MAKAK catalog filepath"""
        cls.KNOWN_CATALOGS['MAKAK']['filepath'] = filepath



class ColorSelector:
    """
    Manages selection of optimal color terms for photometric fitting.
    Chooses colors based on filter wavelengths and target wavelength.
    """

    def __init__(self, filters: Dict[str, CatalogFilter]):
        """
        Initialize with available filters.

        Args:
            filters: Dictionary mapping filter names to CatalogFilter objects
        """
        self.filters = filters
        # Sort filters by wavelength for efficient searching
        self.sorted_filters = sorted(
            filters.items(),
            key=lambda x: x[1].effective_wl
        )

    def select_colors(self, target_wavelength: float, max_colors: int = 4) -> List[dict]:
        """
        Select optimal color pairs for photometric fitting.

        Args:
            target_wavelength: Wavelength of the filter being calibrated (Angstroms)
            max_colors: Maximum number of colors to return (default 4)

        Returns:
            List of dicts containing color information:
            [
                {
                    'blue_filter': 'filter1',
                    'red_filter': 'filter2',
                    'effective_wl': float,  # mean wavelength of the color
                    'width': float,         # wavelength span of the color
                    'description': 'filter1-filter2'
                },
                ...
            ]
        """
        colors = []
        used_filters = set()

        # First, find the closest filter to target wavelength
        base_filter = min(self.filters.items(),
                         key=lambda x: abs(x[1].effective_wl - target_wavelength))
        used_filters.add(base_filter[0])

        # Split remaining filters into blue and red of target
        blue_filters = [
            f for f in self.sorted_filters
            if f[1].effective_wl < target_wavelength and f[0] not in used_filters
        ]
        red_filters = [
            f for f in self.sorted_filters
            if f[1].effective_wl > target_wavelength and f[0] not in used_filters
        ]

        # Strategy 1: Pairs spanning across target wavelength
        span_pairs = self._make_spanning_pairs(blue_filters, red_filters,
                                             target_wavelength)

        # Strategy 2: Neighboring pairs near target wavelength
        neighbor_pairs = self._make_neighbor_pairs(self.sorted_filters,
                                                 target_wavelength)

        # Combine and sort by relevance score
        all_pairs = span_pairs + neighbor_pairs
        all_pairs.sort(key=lambda x: self._score_color_pair(x, target_wavelength))

        # Select best pairs up to max_colors
        selected_pairs = []
        used_filters = set()

        for pair in all_pairs:
            if len(selected_pairs) >= max_colors:
                break

            # Avoid reusing filters unless necessary
            if (pair['blue_filter'] in used_filters or
                pair['red_filter'] in used_filters):
                continue

            selected_pairs.append(pair)
            used_filters.add(pair['blue_filter'])
            used_filters.add(pair['red_filter'])

        # If we still need more colors and have exhausted non-overlapping pairs,
        # allow filter reuse
        if len(selected_pairs) < max_colors:
            for pair in all_pairs:
                if len(selected_pairs) >= max_colors:
                    break
                if pair not in selected_pairs:
                    selected_pairs.append(pair)

        return selected_pairs

    def _make_spanning_pairs(self, blue_filters, red_filters, target_wavelength):
        """Create color pairs that span across the target wavelength"""
        pairs = []

        for blue in blue_filters:
            for red in red_filters:
                blue_wl = blue[1].effective_wl
                red_wl = red[1].effective_wl

                # Skip if the pair doesn't effectively span the target
                if red_wl - blue_wl < 100:  # Minimum wavelength difference in Angstroms
                    continue

                pairs.append({
                    'blue_filter': blue[0],
                    'red_filter': red[0],
                    'effective_wl': (blue_wl + red_wl) / 2,
                    'width': red_wl - blue_wl,
                    'description': f"{blue[0]}-{red[0]}"
                })

        return pairs

    def _make_neighbor_pairs(self, sorted_filters, target_wavelength):
        """Create color pairs from neighboring filters"""
        pairs = []

        for i in range(len(sorted_filters) - 1):
            blue = sorted_filters[i]
            red = sorted_filters[i + 1]

            blue_wl = blue[1].effective_wl
            red_wl = red[1].effective_wl

            pairs.append({
                'blue_filter': blue[0],
                'red_filter': red[0],
                'effective_wl': (blue_wl + red_wl) / 2,
                'width': red_wl - blue_wl,
                'description': f"{blue[0]}-{red[0]}"
            })

        return pairs

    def _score_color_pair(self, pair: dict, target_wavelength: float) -> float:
        """
        Score a color pair based on its relevance to the target wavelength.
        Lower scores are better.

        Scoring considers:
        1. How well the color spans the target wavelength
        2. The width of the color (wider is generally better)
        3. Distance of the color's effective wavelength from target
        """
        effective_wl = pair['effective_wl']
        width = pair['width']

        # Distance from target (normalized by typical filter width ~1000A)
        target_distance = abs(effective_wl - target_wavelength) / 1000.0

        # Width score (normalized to prefer widths around 1000A)
        width_score = abs(width - 1000) / 1000.0

        # Spanning score (better if color spans target wavelength)
        blue_wl = self.filters[pair['blue_filter']].effective_wl
        red_wl = self.filters[pair['red_filter']].effective_wl
        spans_target = (blue_wl <= target_wavelength <= red_wl)
        spanning_score = 0 if spans_target else 1

        # Combine scores (lower is better)
        return target_distance + width_score + spanning_score

    def prepare_color_terms(self, cat: astropy.table.Table,
                          target_filter: str) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare color arrays for photometric fitting.

        Args:
            cat: Catalog data
            target_filter: Name of the filter being calibrated

        Returns:
            Tuple of (color_array, color_descriptions) where:
            - color_array is shape (4, n_stars) containing color values
            - color_descriptions is list of 4 strings describing each color
        """
        if target_filter not in self.filters:
            raise ValueError(f"Unknown target filter: {target_filter}")

        target_wl = self.filters[target_filter].effective_wl
        selected_colors = self.select_colors(target_wl)

        # Prepare output arrays
        n_stars = len(cat)
        color_array = np.zeros((4, n_stars))
        descriptions = []

        # Compute selected colors
        for i, color in enumerate(selected_colors):
            if i >= 4:  # Maximum 4 colors
                break

            blue_mag = cat[self.filters[color['blue_filter']].name]
            red_mag = cat[self.filters[color['red_filter']].name]

            color_array[i] = blue_mag - red_mag
            descriptions.append(color['description'])

        # Pad with zeros and descriptions if we have fewer than 4 colors
        while len(descriptions) < 4:
            descriptions.append('unused')

        return color_array, descriptions

# Example usage in CatalogManager:
def transform_to_instrumental(self, cat, det, wcs):
    """Transform catalog magnitudes to instrumental system"""
    catalog = cat.meta.get('catalog', 'ATLAS')
    filters = self.KNOWN_CATALOGS[catalog]['filters']

    # Create color selector
    selector = ColorSelector(filters)

    # Get target filter
    target_filter = det.meta.get('REFILTER')
    if not target_filter:
        raise ValueError("No target filter specified in detection metadata")

    # Prepare colors
    colors, color_descriptions = selector.prepare_color_terms(cat, target_filter)

    # Store color information in output metadata
    cat_out = cat.copy()
    cat_out.meta['color_terms'] = color_descriptions

    # Transform coordinates
    cat_x, cat_y = wcs.all_world2pix(cat['radeg'], cat['decdeg'], 1)

    # Load and apply photometric model
    ffit = fotfit.fotfit()
    ffit.from_oneline(det.meta['RESPONSE'])

    # Prepare model input
    model_input = (
        cat[filters[target_filter].name],
        det.meta['AIRMASS'],
        (cat_x - det.meta['CTRX'])/1024,
        (cat_y - det.meta['CTRY'])/1024,
        colors[0],
        colors[1],
        colors[2],
        colors[3],
        det.meta['IMGNO'],
        np.zeros_like(cat_x),
        np.ones_like(cat_x)
    )

    cat_out['mag_instrument'] = ffit.model(ffit.fixvalues, model_input)
    return cat_out

def add_catalog_argument(parser):
    """Add catalog selection argument to argument parser"""
    parser.add_argument(
        "--catalog",
        choices=CatalogManager.KNOWN_CATALOGS.keys(),
        default="ATLAS",
        help="Catalog to use for photometric reference"
    )

def run_tests():
    """Test catalog functionality when module is run directly"""
    import time
    from astropy.coordinates import SkyCoord
    import sys

    def print_section(title):
        print("\n" + "="*80)
        print(title)
        print("="*80)

    def print_result(name, success):
        status = "\033[92mPASS\033[0m" if success else "\033[91mFAIL\033[0m"
        print(f"{name:40s} [{status}]")

    def test_catalog_retrieval(catman, name, coords, width, mlim):
        """Test retrieval from a specific catalog"""
        start_time = time.time()
        try:
            cat = catman.get_catalog(coords.ra.deg, coords.dec.deg,
                                   width=width, height=width,
                                   mlim=mlim, catalog=name)
            duration = time.time() - start_time

            if cat is None:
                print(f"  {name:15s} - No data retrieved")
                return False

            print(f"  {name:15s} - Retrieved {len(cat)} objects in {duration:.1f}s")
            print(f"    Available filters: {list(cat.meta.get('filters', []))}")
            print(f"    First object: RA={cat['radeg'][0]:.6f}, "
                  f"Dec={cat['decdeg'][0]:.6f}")

            # Test basic catalog properties
            success = all([
                'radeg' in cat.columns,
                'decdeg' in cat.columns,
                'pmra' in cat.columns,
                'pmdec' in cat.columns,
                len(cat) > 0
            ])

            return success

        except Exception as e:
            print(f"  {name:15s} - Error: {str(e)}")
            return False

    try:
        print_section("Catalog Access Tests")

        # Initialize CatalogManager
        catman = CatalogManager()
        print(f"Atlas directory: {catman.atlas_dir}")
        print("Available catalogs:", list(catman.KNOWN_CATALOGS.keys()))

        # Test coordinates - M31
        coords = SkyCoord('00h42m44.3s +41d16m09s')
        print(f"\nTesting with coordinates: RA={coords.ra.deg:.6f}, "
              f"Dec={coords.dec.deg:.6f} (M31)")

        # Test each catalog
        results = {}
        for cat_name in catman.KNOWN_CATALOGS.keys():
            results[cat_name] = test_catalog_retrieval(
                catman, cat_name, coords, width=0.1, mlim=18.0
            )

        if False:
            print("\nFilter Information Test")
            for cat_name, cat_info in catman.KNOWN_CATALOGS.items():
                filters = cat_info['filters']
                print(f"\n{cat_name} filters:")
                for fname, finfo in filters.items():
                    print(f"  {fname:10s}: λ={finfo.effective_wl:5.0f}Å, "
                          f"system={finfo.system:4s}")

            print_section("Color Selection Tests")

            # Test color selection logic
            test_filters = {
                'Johnson_V': 5500,
                'Sloan_r': 6200,
                'Sloan_i': 7500,
            }

            for filter_name, wavelength in test_filters.items():
                print(f"\nTesting color selection for {filter_name} "
                      f"(λ={wavelength}Å):")

                for cat_name, cat_info in catman.KNOWN_CATALOGS.items():
                    selector = ColorSelector(cat_info['filters'])
                    colors = selector.select_colors(wavelength)
                    print(f"\n{cat_name}:")
                    print("  Selected colors:", colors)

        print_section("Test Results Summary")

        all_passed = True
        for cat_name, result in results.items():
            print_result(cat_name, result)
            all_passed = all_passed and result

        if all_passed:
            print("\nAll tests passed successfully!")
            return 0
        else:
            print("\nSome tests failed - check output above")
            return 1

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    # Initialize with error handling for astroquery imports
    try:
        from astroquery.vizier import Vizier
        from astroquery.gaia import Gaia
        from astroquery.mast import Catalogs
        online_catalogs_available = True
    except ImportError:
        warnings.warn("astroquery not available - online catalogs disabled")
        online_catalogs_available = False

    sys.exit(run_tests())
