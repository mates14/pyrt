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

class Catalog(astropy.table.Table):
    """
    Represents a stellar catalog with methods for retrieval and transformation.
    Inherits from astropy Table while providing catalog management functionality.
    """

    # Catalog identifiers
    ATLAS = 'atlas@localhost'
    ATLAS_VIZIER = 'atlas@vizier'
    PANSTARRS = 'panstarrs'
    GAIA = 'gaia'
    MAKAK = 'makak'

    # Define available catalogs with their properties
    KNOWN_CATALOGS = {
        ATLAS: {
            'description': 'Local ATLAS catalog',
            'filters': CatalogFilters.ATLAS,
            'epoch': 2015.5,
            'local': True,
            'service': 'local',
            'mag_splits': [
                ('00_m_16', 0),
                ('16_m_17', 16),
                ('17_m_18', 17),
                ('18_m_19', 18),
                ('19_m_20', 19)
            ]
        },
        PANSTARRS: {
            'description': 'Pan-STARRS Data Release 2',
            'filters': CatalogFilters.PANSTARRS,
            'catalog_id': 'Panstarrs', #?
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
        GAIA: {
            'description': 'Gaia Data Release 3',
            'filters': CatalogFilters.GAIA,
            'epoch': 2016.0,
            'local': False,
            'service': 'Gaia',
            'catalog_id': 'gaiadr3.gaia_source'
        },
        ATLAS_VIZIER: {
            'description': 'ATLAS Reference Catalog 2',
            'filters': CatalogFilters.ATLAS,
            'epoch': 2015.5,
            'local': False,
            'service': 'VizieR',
            'catalog_id': 'J/ApJ/867/105',  # Updated catalog reference
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
        MAKAK: {
            'description': 'Pre-filtered wide-field catalog',
            'filters': CatalogFilters.ATLAS,  # Using ATLAS filter definitions
            'epoch': 2015.5,  # Default epoch, could be overridden from FITS metadata
            'local': True,
            'service': 'local',
            'filepath': '/home/mates/test/catalog.fits'  # Default path, could be configurable
        }
    }

    def __init__(self, *args, **kwargs):
        """Initialize the catalog."""
        # Extract catalog query parameters if provided
        self.ra = kwargs.pop('ra', None)
        self.dec = kwargs.pop('dec', None)
        self.width = kwargs.pop('width', 0.25)
        self.height = kwargs.pop('height', 0.25)
        self.mlim = kwargs.pop('mlim', 17.0)
        self.catalog_name = kwargs.pop('catalog', None)
        self.timeout = kwargs.pop('timeout', 60)
        self.atlas_dir = kwargs.pop('atlas_dir', "/home/mates/cat/atlas")

        # If catalog name provided, fetch data
        if self.catalog_name:
            result = self._fetch_catalog_data()
            super().__init__(result, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def _fetch_catalog_data(self):
        """Fetch data from the specified catalog source"""
        if self.catalog_name not in self.KNOWN_CATALOGS:
            raise ValueError(f"Unknown catalog: {self.catalog_name}")

        config = self.KNOWN_CATALOGS[self.catalog_name]
        result = None

        # Get catalog data
        if self.catalog_name == self.ATLAS:
            result = self._get_atlas_local()
        elif self.catalog_name == self.ATLAS_VIZIER:
            result = self._get_atlas_vizier()
        elif self.catalog_name == self.PANSTARRS:
            result = self._get_panstarrs_data()
        elif self.catalog_name == self.GAIA:
            result = self._get_gaia_data()
        elif self.catalog_name == self.MAKAK:
            result = self._get_makak_data()
        else:
            raise ValueError(f"Unknown catalog: {catalog}")

        if result is None:
            raise ValueError(f"No data retrieved from {self.catalog_name}")

        result.meta.update({
            'catalog': self.catalog_name,
            'astepoch': config['epoch'],
            'filters': list(config['filters'].keys())
        })
        return result

    def _get_atlas_local(self):
        """Get data from local ATLAS catalog"""
        config = self.KNOWN_CATALOGS[self.ATLAS]
        result = None

        for dirname, magspl in config['mag_splits']:
            if self.mlim <= magspl:
                continue

            print(f"get_atlas: mlim:{self.mlim} > magspl:{magspl}")

            directory = os.path.join(self.atlas_dir, dirname)
            try:
                new_data = self._get_atlas_split(directory)
                if new_data is None:
                    continue

                result = new_data if result is None else astropy.table.vstack([result, new_data])

            except Exception as e:
                warnings.warn(f"get_atlas: Failed to get data from {directory}: {e}")

        if result is not None and len(result) > 0:
            self._add_transformed_magnitudes(result)

        print(f"get_atlas: returning {len(result)} records")
        return result

    def _get_atlas_split(self, directory):
        """Get data from one magnitude split of ATLAS catalog"""
        atlas_ecsv_tmp = f"atlas{os.getpid()}.ecsv"
        try:
            cmd = f'atlas {self.ra} {self.dec} -rect {self.width},{self.height} '\
                  f'-dir {directory} -mlim {self.mlim:.2f} -ecsv > {atlas_ecsv_tmp}'
            os.system(cmd)
            return astropy.table.Table.read(atlas_ecsv_tmp, format='ascii.ecsv')
        finally:
            with suppress(FileNotFoundError):
                os.remove(atlas_ecsv_tmp)

    @staticmethod
    def _add_transformed_magnitudes(cat):
        """Add transformed Johnson magnitudes"""
        gr = cat['Sloan_g'] - cat['Sloan_r']
        ri = cat['Sloan_r'] - cat['Sloan_i']
        iz = cat['Sloan_i'] - cat['Sloan_z']

        cat['Johnson_B'] = cat['Sloan_r'] + 1.490989 * gr + \
                          0.125787 * gr * gr - 0.022359 * gr*gr*gr + 0.186304
        cat['Johnson_V'] = cat['Sloan_r'] + 0.510236 * gr - 0.0337082
        cat['Johnson_R'] = cat['Sloan_r'] - 0.197420 * ri - \
                          0.083113 * ri * ri - 0.179943
        cat['Johnson_I'] = cat['Sloan_r'] - 0.897087 * ri - \
                          0.575316 * iz - 0.423971

    def _get_atlas_vizier(self):
        """Get ATLAS RefCat2 data from VizieR with updated column mapping"""
 #       try:
        if True:
            from astroquery.vizier import Vizier
            # Configure Vizier with correct column names
            column_mapping = self.KNOWN_CATALOGS[self.ATLAS_VIZIER]['column_mapping']
            vizier = Vizier(
                columns=list(column_mapping.keys()),
                column_filters={
                    "rmag": f"<{self.mlim}"  # Magnitude limit in r-band
                },
                row_limit=-1
            )

            # Create coordinate object
            coords = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg, frame='icrs')

            # Query VizieR
            result = vizier.query_region(
                coords,
                width=self.width * u.deg,
                height=self.height * u.deg,
                catalog=self.KNOWN_CATALOGS[self.ATLAS_VIZIER]['catalog_id']
            )

            if not result or len(result) == 0:
                return None

            atlas = result[0]

            # Create output catalog
            cat = astropy.table.Table(result)

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

            return cat
            if result is not None and len(result) > 0:
                self._add_transformed_magnitudes(result)

            print(f"get_atlas: returning {len(result)} records")
            return result


#        except Exception as e:
#            warnings.warn(f"VizieR ATLAS query failed: {e}")
#            return None

    def _get_panstarrs_data(self):
        """Get PanSTARRS DR2 data"""
#        try:
        if True:
            from astroquery.mast import Catalogs

            config = self.KNOWN_CATALOGS[self.PANSTARRS]
            radius = np.sqrt(self.width**2 + self.height**2) / 2
            coords = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg, frame='icrs')

            constraints = {
                'nDetections.gt': 4,
                'rMeanPSFMag.lt': self.mlim,
                'qualityFlag.lt': 128
            }

            ps1 = Catalogs.query_region(
                coords,
                catalog=config['catalog_id'],
                radius=radius * u.deg,
                data_release="dr2",
                table=config['table'],
                **constraints
            )

            if len(ps1) == 0:
                return None

            result = astropy.table.Table()

            # Map columns according to configuration
            for ps1_name, our_name in config['column_mapping'].items():
                if ps1_name in ps1.columns:
                    result[our_name] = ps1[ps1_name].astype(np.float64)

            # Add proper motion columns (not provided by PanSTARRS)
            result['pmra'] = np.zeros(len(ps1), dtype=np.float64)
            result['pmdec'] = np.zeros(len(ps1), dtype=np.float64)

            return result

#        except Exception as e:
#            raise ValueError(f"PanSTARRS query failed: {str(e)}")

    def _get_gaia_data(self):
        """Get Gaia DR3 data"""
        try:
            from astroquery.gaia import Gaia

            config = self.KNOWN_CATALOGS[self.GAIA]
            query = f"""
            SELECT
                source_id, ra, dec, pmra, pmdec,
                phot_g_mean_mag, phot_g_mean_flux_over_error,
                phot_bp_mean_mag, phot_bp_mean_flux_over_error,
                phot_rp_mean_mag, phot_rp_mean_flux_over_error
            FROM {config['catalog_id']}
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                BOX('ICRS', {self.ra}, {self.dec}, {self.width}, {self.height}))
                AND phot_g_mean_mag < {self.mlim}
                AND ruwe < 1.4
                AND visibility_periods_used >= 8
            """

            job = Gaia.launch_job_async(query)
            gaia_cat = job.get_results()

            if len(gaia_cat) == 0:
                return None

            result = astropy.table.Table()

            # Basic astrometry
            result['radeg'] = gaia_cat['ra']
            result['decdeg'] = gaia_cat['dec']
            result['pmra'] = gaia_cat['pmra'] / (3.6e6)  # mas/yr to deg/yr
            result['pmdec'] = gaia_cat['pmdec'] / (3.6e6)  # mas/yr to deg/yr

            # Add Gaia magnitudes and errors
            for filter_name, filter_info in config['filters'].items():
                result[filter_info.name] = gaia_cat[filter_info.name]
                if filter_info.error_name:
                    flux_over_error = gaia_cat[filter_info.name.replace('mag', 'flux_over_error')]
                    result[filter_info.error_name] = 2.5 / (flux_over_error * np.log(10))

            return result

        except Exception as e:
            raise ValueError(f"Gaia query failed: {str(e)}")

    def _get_makak_data(self):
        """Get data from pre-filtered MAKAK catalog"""
        try:
            config = self.KNOWN_CATALOGS[self.MAKAK]

            # Read the pre-filtered catalog
            cat = astropy.table.Table.read(config['filepath'])

            # Filter by field of view
            ctr = SkyCoord(self.ra*u.deg, self.dec*u.deg, frame='fk5')
            cnr = SkyCoord((self.ra+self.width)*u.deg, (self.dec+self.height)*u.deg, frame='fk5')
            radius = cnr.separation(ctr) / 2

            cat_coords = SkyCoord(cat['radeg']*u.deg, cat['decdeg']*u.deg, frame='fk5')
            within_field = cat_coords.separation(ctr) < radius
            cat = cat[within_field]

            if len(cat) == 0:
                return None

            # Ensure proper motion columns exist
            if 'pmra' not in cat.columns:
                cat['pmra'] = np.zeros(len(cat), dtype=np.float64)
            if 'pmdec' not in cat.columns:
                cat['pmdec'] = np.zeros(len(cat), dtype=np.float64)

            return cat

        except Exception as e:
            raise ValueError(f"MAKAK catalog access failed: {str(e)}")

    @classmethod
    def from_file(cls, filename):
        """Create catalog instance from a local file."""
        try:
            data = astropy.table.Table.read(filename)
            obj = cls(data.as_array(), meta=data.meta)
            obj.meta['catalog'] = 'local'
            return obj
        except Exception as e:
            raise ValueError(f"Failed to read catalog from {filename}: {str(e)}")

    @property
    def description(self):
        """Get catalog description."""
        if self.catalog_name in self.KNOWN_CATALOGS:
            return self.KNOWN_CATALOGS[self.catalog_name].description
        return "Unknown catalog"

    @property
    def filters(self):
        """Get available filters."""
        if self.catalog_name in self.KNOWN_CATALOGS:
            return self.KNOWN_CATALOGS[self.catalog_name].filters
        return {}

    @property
    def epoch(self):
        """Get catalog epoch."""
        if self.catalog_name in self.KNOWN_CATALOGS:
            return self.KNOWN_CATALOGS[self.catalog_name].epoch
        return None

    def transform_to_instrumental(self, det, wcs):
        """
        Transform catalog to instrumental system.

        Args:
            det: Detection metadata table
            wcs: WCS for coordinate transformation

        Returns:
            Catalog: New catalog instance with transformed data
        """
        try:
            # Get target filter
            target_filter = det.meta.get('REFILTER')
            if not target_filter:
                raise ValueError("No target filter (REFILTER) specified in detection metadata")

            # Create color selector and get colors
            selector = ColorSelector(self.filters)
            colors, color_descriptions = selector.prepare_color_terms(self, target_filter)

            # Create output catalog
            cat_out = self.copy()
            cat_out.meta['color_terms'] = color_descriptions
            cat_out.meta['target_filter'] = target_filter

            # Transform coordinates
            try:
                cat_x, cat_y = wcs.all_world2pix(self['radeg'], self['decdeg'], 1)
            except Exception as e:
                raise ValueError(f"Coordinate transformation failed: {str(e)}")

            # Load photometric model
            if 'RESPONSE' not in det.meta:
                raise ValueError("No RESPONSE model in detection metadata")

            try:
                import fotfit
                ffit = fotfit.FotFit()
                ffit.from_oneline(det.meta['RESPONSE'])
            except Exception as e:
                raise ValueError(f"Failed to load photometric model: {str(e)}")

            # Get base magnitude
            filter_info = self.filters[target_filter]
            base_mag = self[filter_info.name]

            # Prepare model input
            model_input = (
                base_mag,
                det.meta['AIRMASS'],
                (cat_x - det.meta['CTRX'])/1024,
                (cat_y - det.meta['CTRY'])/1024,
                colors[0], colors[1], colors[2], colors[3],
                det.meta['IMGNO'],
                np.zeros_like(base_mag),
                np.ones_like(base_mag)
            )

            # Apply model
            cat_out['mag_instrument'] = ffit.model(ffit.fixvalues, model_input)

            # Add errors
            if filter_info.error_name and filter_info.error_name in self.columns:
                cat_out['mag_instrument_err'] = np.sqrt(
                    self[filter_info.error_name]**2 + 0.01**2
                )
            else:
                cat_out['mag_instrument_err'] = np.full_like(base_mag, 0.03)

            # Add transformation metadata
            cat_out.meta['transform_info'] = {
                'source_catalog': self.catalog_name,
                'source_filter': filter_info.name,
                'target_filter': target_filter,
                'color_terms': color_descriptions,
                'airmass': float(det.meta['AIRMASS']),
                'model': det.meta['RESPONSE']
            }

            return cat_out

        except Exception as e:
            raise ValueError(f"Transformation failed: {str(e)}")

class ColorSelector:
    """
    Manages selection of optimal color terms for photometric fitting.
    Chooses colors based on filter wavelengths and target wavelength.
    """

    def __init__(self, filters: Dict[str, CatalogFilter]):
        """Initialize with available filters."""
        self.filters = filters
        self.sorted_filters = sorted(
            filters.items(),
            key=lambda x: x[1].effective_wl
        )

    def select_colors(self, target_wavelength: float, max_colors: int = 4) -> List[dict]:
        """Select optimal color pairs for photometric fitting."""
        colors = []
        used_filters = set()

        # Find closest filter to target
        base_filter = min(
            self.filters.items(),
            key=lambda x: abs(x[1].effective_wl - target_wavelength)
        )
        used_filters.add(base_filter[0])

        # Split remaining filters by wavelength
        blue_filters = [
            f for f in self.sorted_filters
            if f[1].effective_wl < target_wavelength
            and f[0] not in used_filters
        ]
        red_filters = [
            f for f in self.sorted_filters
            if f[1].effective_wl > target_wavelength
            and f[0] not in used_filters
        ]

        # Get color pairs
        span_pairs = self._make_spanning_pairs(blue_filters, red_filters, target_wavelength)
        neighbor_pairs = self._make_neighbor_pairs(self.sorted_filters, target_wavelength)

        # Combine and sort
        all_pairs = span_pairs + neighbor_pairs
        all_pairs.sort(key=lambda x: self._score_color_pair(x, target_wavelength))

        # Select best pairs
        selected = []
        used = set()

        # First try without filter reuse
        for pair in all_pairs:
            if len(selected) >= max_colors:
                break

            if pair['blue_filter'] in used or pair['red_filter'] in used:
                continue

            selected.append(pair)
            used.add(pair['blue_filter'])
            used.add(pair['red_filter'])

        # Allow filter reuse if needed
        if len(selected) < max_colors:
            for pair in all_pairs:
                if len(selected) >= max_colors:
                    break
                if pair not in selected:
                    selected.append(pair)

        return selected

    def prepare_color_terms(self, cat: astropy.table.Table, target_filter: str) -> Tuple[np.ndarray, List[str]]:
        """Prepare color arrays for photometric fitting."""
        if target_filter not in self.filters:
            raise ValueError(f"Unknown target filter: {target_filter}")

        target_wl = self.filters[target_filter].effective_wl
        selected_colors = self.select_colors(target_wl)

        n_stars = len(cat)
        color_array = np.zeros((4, n_stars))
        descriptions = []

        for i, color in enumerate(selected_colors):
            if i >= 4:
                break

            blue_mag = cat[self.filters[color['blue_filter']].name]
            red_mag = cat[self.filters[color['red_filter']].name]

            color_array[i] = blue_mag - red_mag
            descriptions.append(color['description'])

        while len(descriptions) < 4:
            descriptions.append('unused')

        return color_array, descriptions

    def _make_spanning_pairs(self, blue_filters, red_filters, target_wavelength):
        """Create color pairs that span the target wavelength."""
        pairs = []
        for blue in blue_filters:
            for red in red_filters:
                blue_wl = blue[1].effective_wl
                red_wl = red[1].effective_wl

                if red_wl - blue_wl < 100:
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
        """Create color pairs from neighboring filters."""
        pairs = []
        for i in range(len(sorted_filters) - 1):
            blue = sorted_filters[i]
            red = sorted_filters[i + 1]

            pairs.append({
                'blue_filter': blue[0],
                'red_filter': red[0],
                'effective_wl': (blue[0].effective_wl + red[1].effective_wl) / 2,
                'width': red[1].effective_wl - blue[1].effective_wl,
                'description': f"{blue[0]}-{red[0]}"
            })
        return pairs

    def _score_color_pair(self, pair: dict, target_wavelength: float) -> float:
        """Score a color pair based on relevance to target wavelength."""
        effective_wl = pair['effective_wl']
        width = pair['width']

        target_distance = abs(effective_wl - target_wavelength) / 1000.0
        width_score = abs(width - 1000) / 1000.0

        blue_wl = self.filters[pair['blue_filter']].effective_wl
        red_wl = self.filters[pair['red_filter']].effective_wl
        spans_target = (blue_wl <= target_wavelength <= red_wl)

        return target_distance + width_score + (0 if spans_target else 1)

def add_catalog_argument(parser):
    """Add catalog selection argument to argument parser"""
    parser.add_argument(
        "--catalog",
        choices=Catalog.KNOWN_CATALOGS.keys(),
        default="ATLAS",
        help="Catalog to use for photometric reference"
    )
