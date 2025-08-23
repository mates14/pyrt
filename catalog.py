#!/usr/bin/python3

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any, Type, TypeVar, cast
import numpy as np
import warnings
import os
import subprocess
import tempfile
import astropy.table
import astropy.io.ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
import logging


# Type aliases
TableType = TypeVar("TableType", bound=astropy.table.Table)
CatalogConfig = Dict[str, Any]
FilterDict = Dict[str, "CatalogFilter"]

@dataclass
class QueryParams:
    """Parameters used for catalog queries"""
    ra: Optional[float] = None
    dec: Optional[float] = None
    width: float = 0.25
    height: float = 0.25
    mlim: float = 17.0
    timeout: int = 60
    atlas_dir: str = "/home/mates/cat/atlas"

@dataclass
class CatalogFilter:
    """Information about a filter in a catalog"""
    name: str           # Original filter name in catalog
    effective_wl: float # Effective wavelength in Angstroms
    system: str        # Photometric system (e.g., 'AB', 'Vega')
    error_name: Optional[str] = None  # Name of error column if available

class CatalogFilters:
    """Filter definitions for different catalogs"""

    PANSTARRS: FilterDict
    GAIA: FilterDict
    ATLAS: FilterDict
    USNOB: FilterDict

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
        # IS THIS VEGA OR AB?
        'G': CatalogFilter('G', 5890, 'AB', 'G_err'),
        'BP': CatalogFilter('BP', 5050, 'AB', 'BP_err'),
        'RP': CatalogFilter('RP', 7730, 'AB', 'RP_err'),
        'Johnson_B': CatalogFilter('Johnson_B', 4353, 'Vega'),
        'Johnson_V': CatalogFilter('Johnson_V', 5477, 'Vega'),
        'Johnson_R': CatalogFilter('Johnson_R', 6349, 'Vega'),
        'Johnson_I': CatalogFilter('Johnson_I', 8797, 'Vega'),
    }
    # ATLAS filters
    ATLAS = {
        'Sloan_g': CatalogFilter('Sloan_g', 4810, 'AB'),
        'Sloan_r': CatalogFilter('Sloan_r', 6170, 'AB'),
        'Sloan_i': CatalogFilter('Sloan_i', 7520, 'AB'),
        'Sloan_z': CatalogFilter('Sloan_z', 8660, 'AB'),
        'J': CatalogFilter('J', 12000, 'AB'),
        'J_Vega': CatalogFilter('J_Vega', 12000, 'Vega'),  # J_AB - 0.894 mag (H_Vega: -1.374, K_Vega: -1.85 from Cohen et al. 2003)
        'Johnson_B': CatalogFilter('Johnson_B', 4353, 'Vega'),
        'Johnson_V': CatalogFilter('Johnson_V', 5477, 'Vega'),
        'Johnson_R': CatalogFilter('Johnson_R', 6349, 'Vega'),
        'Johnson_I': CatalogFilter('Johnson_I', 8797, 'Vega'),
    }
    USNOB = {
        'B1': CatalogFilter('B1', 4500, 'AB', 'e_B1mag'),
        'R1': CatalogFilter('R1', 6400, 'AB', 'e_R1mag'),
        'B2': CatalogFilter('B2', 4500, 'AB', 'e_B2mag'),
        'R2': CatalogFilter('R2', 6400, 'AB', 'e_R2mag'),
        'I': CatalogFilter('I', 8100, 'AB', 'e_Imag'),
    }

class Catalog(astropy.table.Table):
    """
    Represents a stellar catalog with methods for retrieval and transformation.
    Inherits from astropy Table while providing catalog management functionality.
    """

    # Catalog identifiers
    ATLAS: str = 'atlas@localhost'
    ATLAS_VIZIER: str = 'atlas@vizier'
    PANSTARRS: str = 'panstarrs'
    GAIA: str = 'gaia'
    MAKAK: str = 'makak'
    USNOB: str = 'usno'

    # Define available catalogs with their properties
    KNOWN_CATALOGS: Dict[str, CatalogConfig]
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
            'catalog_id': 'gaiadr3.gaia_source',
            'column_mapping': {
                'phot_g_mean_mag': 'G',
                'phot_g_mean_mag_error': 'G_err',
                'phot_bp_mean_mag': 'BP',
                'phot_bp_mean_mag_error': 'BP_err',
                'phot_rp_mean_mag': 'RP',
                'phot_rp_mean_mag_error': 'RP_err',
            }
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
            'description': 'Pre-filtered Gaia bright stars catalog',
            'filters': CatalogFilters.GAIA,
            'epoch': 2016.0,
            'local': True,
            'service': 'local',
            'filepath': '/home/mates/gaia_bright_stars.fits'
        },
        USNOB: {
            'description': 'USNO-B1.0 Catalog',
            'filters': CatalogFilters.USNOB,
            'epoch': 2000.0,
            'local': False,
            'service': 'VizieR',
            'catalog_id': 'I/284/out',
            'column_mapping': {
                'RAJ2000': 'radeg',
                'DEJ2000': 'decdeg',
                'B1mag': 'B1',
                'R1mag': 'R1',
                'B2mag': 'B2',
                'R2mag': 'R2',
                'Imag': 'I',
                'e_B1mag': 'B1_err',
                'e_R1mag': 'R1_err',
                'e_B2mag': 'B2_err',
                'e_R2mag': 'R2_err',
                'e_Imag': 'I_err',
                'pmRA': 'pmra',
                'pmDE': 'pmdec'
            }
        }
    }

    def __init__(self, *args, **kwargs):
        """Initialize the catalog with proper handling of properties."""
        # Extract and store query parameters
        query_params: Dict[str, Any] = {}
        for param in QueryParams.__dataclass_fields__:
            if param in kwargs:
                query_params[param] = kwargs.pop(param)
        self._query_params: QueryParams = QueryParams(**query_params)

        # Store catalog name and get config
        self._catalog_name: Optional[str] = kwargs.pop("catalog", None)
        self._config: CatalogConfig = (self.KNOWN_CATALOGS[self._catalog_name]
                       if self._catalog_name in self.KNOWN_CATALOGS
                       else {})

        # Initialize base Table
        if self._catalog_name:
            result = self._fetch_catalog_data()
            super().__init__(result, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

        # Ensure catalog metadata is properly stored
        self._init_metadata()

    def _init_metadata(self) -> None:
        """Initialize or update catalog metadata"""
        if 'catalog_props' not in self.meta:
            self.meta['catalog_props'] = {}

        catalog_props: Dict[str, Any] = {
            'catalog_name': self._catalog_name,
            'query_params': asdict(self._query_params) if self._query_params else None,
            'epoch': self._config.get('epoch'),
            'filters': {k: asdict(v) for k, v in self._config.get('filters', {}).items()},
            'description': self._config.get('description'),
        }

        self.meta['catalog_props'].update(catalog_props)

    @property
    def query_params(self) -> Optional[QueryParams]:
        """Get query parameters used to create this catalog"""
        params_dict = self.meta.get('catalog_props', {}).get('query_params', {})
        return QueryParams(**params_dict) if params_dict else None

    @property
    def catalog_name(self) -> str:
        """Get catalog name"""
        return str(self.meta.get('catalog_props', {}).get('catalog_name'))

    def _fetch_catalog_data(self) -> Optional[astropy.table.Table]:
        """Fetch data from the specified catalog source"""
        if self._catalog_name not in self.KNOWN_CATALOGS.keys():
            raise ValueError(f"Unknown catalog: {self._catalog_name}")

        config = self.KNOWN_CATALOGS[self._catalog_name]
        result: Optional[astropy.table.Table] = None
 
        # Get catalog data
        if self._catalog_name == self.ATLAS:
            result = self._get_atlas_local()
        elif self._catalog_name == self.ATLAS_VIZIER:
            result = self._get_atlas_vizier()
        elif self._catalog_name == self.PANSTARRS:
            result = self._get_panstarrs_data()
        elif self._catalog_name == self.GAIA:
            result = self._get_gaia_data()
        elif self._catalog_name == self.MAKAK:
            result = self._get_makak_data()
        elif self._catalog_name == self.USNOB:
            result = self._get_usnob_data()
        else:
            raise ValueError(f"Unknown catalog: {self._catalog_name}")

        if result is None:
            raise ValueError(f"No data retrieved from {self._catalog_name}")

        result.meta.update({
            'catalog': self._catalog_name,
            'astepoch': config['epoch'],
            'filters': list(config['filters'].keys())
        })
        return result

    def _get_atlas_local(self) -> Optional[astropy.table.Table]:
        """Get data from local ATLAS catalog"""
        config = self.KNOWN_CATALOGS[self.ATLAS]
        result: Optional[astropy.table.Table] = None

        for dirname, magspl in config['mag_splits']:
            if self._query_params.mlim <= magspl:
                continue

            directory = os.path.join(self._query_params.atlas_dir, dirname)
            new_data = self._get_atlas_split(directory)
            if new_data is None:
                continue

            result = new_data if result is None else astropy.table.vstack([result, new_data])

        if result is not None and len(result) > 0:
            self._add_transformed_magnitudes(result)

        return result

    def _get_atlas_split(self, directory: str) -> Optional[astropy.table.Table]:
        """Get data from one magnitude split of ATLAS catalog"""
        with tempfile.NamedTemporaryFile(suffix=".ecsv", delete=False) as tmp:
            try:
                cmd =   f'atlas {self._query_params.ra} {self._query_params.dec} '\
                        f'-rect {self._query_params.width},{self._query_params.height} '\
                        f'-dir {directory} -mlim {self._query_params.mlim:.2f} -ecsv '
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                with open(tmp.name, "w") as f:
                    f.write(result.stdout)

                return astropy.table.Table.read(tmp.name, format="ascii.ecsv")

            finally:
                os.unlink(tmp.name)

    @staticmethod
    def _add_transformed_magnitudes(cat: astropy.table.Table) -> None:
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

        # Add J_Vega for Johnson system compatibility
        # Using 2MASS standard AB→Vega conversion: J(Vega) = J(AB) - 0.894
        # Reference: Cohen et al. (2003) AJ 126, 1090 - "Spectral Irradiance Calibration in the Infrared. XIV"
        cat['J_Vega'] = cat['J'] - 0.894

    def _get_atlas_vizier(self) -> Optional[astropy.table.Table]:
        """Get ATLAS RefCat2 data from VizieR with updated column mapping"""
        from astroquery.vizier import Vizier
        # Configure Vizier with correct column names
        column_mapping = self.KNOWN_CATALOGS[self.ATLAS_VIZIER]['column_mapping']
        vizier = Vizier(
            columns=list(column_mapping.keys()),
            column_filters={
                "rmag": f"<{self._query_params.mlim}"  # Magnitude limit in r-band
            },
            row_limit=-1
        )

        # Create coordinate object
        coords = SkyCoord(
            ra=self._query_params.ra*u.deg,
            dec=self._query_params.dec*u.deg,
            frame='icrs'
        )

        # Query VizieR
        result = vizier.query_region(
            coords,
            width=self._query_params.width * u.deg,
            height=self._query_params.height * u.deg,
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

    def _get_panstarrs_data(self) -> Optional[astropy.table.Table]:
        """Get PanSTARRS DR2 data"""
        from astroquery.mast import Catalogs

        config = self.KNOWN_CATALOGS[self.PANSTARRS]
        radius = np.sqrt(self._query_params.width**2 + self._query_params.height**2) / 2
        coords = SkyCoord(ra=self._query_params.ra*u.deg, dec=self._query_params.dec*u.deg, frame='icrs')

        constraints = {
            'nDetections.gt': 4,
            'rMeanPSFMag.lt': self._query_params.mlim,
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

    def _get_gaia_data(self) -> Optional[astropy.table.Table]:
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
                BOX('ICRS', {self._query_params.ra}, {self._query_params.dec},
                    {2*self._query_params.width}, {2*self._query_params.height}))
                AND phot_g_mean_mag < {self._query_params.mlim}
                AND ruwe < 1.4
                AND visibility_periods_used >= 8
                -- Ensure we only get complete photometric data
                AND phot_g_mean_mag IS NOT NULL
                AND phot_bp_mean_mag IS NOT NULL
                AND phot_rp_mean_mag IS NOT NULL
                AND phot_g_mean_flux_over_error > 0
                AND phot_bp_mean_flux_over_error > 0
                AND phot_rp_mean_flux_over_error > 0
            """

            job = Gaia.launch_job_async(query)
            gaia_cat = job.get_results()

            if len(gaia_cat) == 0:
                return None

            result = astropy.table.Table()

            # Basic astrometry
            result['radeg'] = gaia_cat['ra']
            result['decdeg'] = gaia_cat['dec']
            try:
                result['pmra'] = np.float64(gaia_cat['pmra']) / (3.6e6)  # mas/yr to deg/yr
            except TypeError:
                result['pmra'] = 0
            try:
                result['pmdec'] = np.float64(gaia_cat['pmdec']) / (3.6e6)  # mas/yr to deg/yr
            except TypeError:
                result['pmdec'] = 0

            # Map columns according to configuration
            for gaia_name, our_name in config['column_mapping'].items():
                if gaia_name in gaia_cat.columns:
                    result[our_name] = gaia_cat[gaia_name].astype(np.float64)

            # Based on Stetson stars
            bpg = result['BP'] - result['G']
            grp = result['G'] - result ['RP']

            # P1D     =        -0.666169 / ± 0.022010 (3.304%)
            # P2D     =        -0.145798 / ± 0.035219 (24.156%)
            # P3D     =        -0.517584 / ± 0.018302 (3.536%)

            result['Johnson_B'] = result['BP'] + 0.0085 \
                +0.666169*grp +0.145798*grp*grp +0.517584*grp*grp*grp

            # P1C     =        -0.368628 / ± 0.014166 (3.843%)
            # P2C     =        -0.139117 / ± 0.004046 (2.908%)
            # P1D     =         0.103787 / ± 0.023852 (22.982%)
            # P2D     =        -0.194751 / ± 0.040714 (20.906%)
            # P3D     =        -0.156161 / ± 0.021977 (14.073%)

            result['Johnson_V'] = result['G'] + 0.0052 \
                +0.368628*bpg +0.139117*bpg*bpg  \
                -0.103787*grp +0.194751*grp*grp +0.156161*grp*grp*grp

            # P1C     =         0.036933 / ± 0.033571 (90.897%)
            # P2C     =         0.083075 / ± 0.020279 (24.410%)
            # P3C     =        -0.022532 / ± 0.003502 (15.544%)
            # P1D     =         0.353593 / ± 0.024744 (6.998%)
            # P2D     =         0.482768 / ± 0.049077 (10.166%)
            # P3D     =        -0.728803 / ± 0.031478 (4.319%)

            result['Johnson_R'] = result['G'] + 0.0167 \
                -0.036933*bpg -0.083075*bpg*bpg +0.022532*bpg*bpg*bpg \
                -0.353593*grp -0.482768*grp*grp +0.728803*grp*grp*grp

            # P1D     =        -0.253449 / ± 0.024239 (9.564%)
            # P2D     =         0.113991 / ± 0.011589 (10.167%)
            # P1C     =         0.395266 / ± 0.021690 (5.488%)
            # P2C     =        -0.183776 / ± 0.011980 (6.519%)
            # P3C     =         0.020945 / ± 0.002219 (10.593%)

            result['Johnson_I'] = result['RP'] - 0.0660 \
                -0.395266*bpg +0.183776*bpg*bpg -0.020945*bpg*bpg*bpg \
                +0.253449*grp -0.113991*grp*grp

            return result

        except Exception as e:
            raise ValueError(f"Gaia query failed: {str(e)}")

    def _get_usnob_data(self) -> Optional[astropy.table.Table]:
        """Get USNO-B1.0 data from VizieR"""
        try:
            from astroquery.vizier import Vizier

            config = self.KNOWN_CATALOGS[self.USNOB]
            column_mapping = config['column_mapping']

            # Configure Vizier
            vizier = Vizier(
                columns=list(column_mapping.keys()),
                column_filters={
                    "R1mag": f"<{self._query_params.mlim}"  # Magnitude limit in R1
                },
                row_limit=-1  # Get all matching objects
            )

            # Create coordinate object
            coords = SkyCoord(
                ra=self._query_params.ra*u.deg,
                dec=self._query_params.dec*u.deg,
                frame='icrs'
            )

            # Query VizieR
            result = vizier.query_region(
                coords,
                width=2*self._query_params.width * u.deg,
                height=2*self._query_params.height * u.deg,
                catalog=config['catalog_id']
            )

            if not result or len(result) == 0:
                logging.warning("No USNO-B data found")
                return None

            usnob = result[0]

            # Create output catalog
            cat = astropy.table.Table()

            # Initialize mapped columns
            our_columns = set(column_mapping.values())
            for col in our_columns:
                cat[col] = np.zeros(len(usnob), dtype=np.float64)

            # Map columns according to configuration
            for vizier_name, our_name in column_mapping.items():
                if vizier_name in usnob.columns:
                    # Convert proper motions from mas/yr to deg/yr if needed
                    if vizier_name in ['pmRA', 'pmDE']:
                        cat[our_name] = usnob[vizier_name] / (3.6e6)
                    else:
                        cat[our_name] = usnob[vizier_name]

            # Handle quality flags and uncertainties
            for band in ['B1', 'R1', 'B2', 'R2', 'I']:
                mag_col = f'{band}mag'
                err_col = f'e_{band}mag'
                if mag_col in cat.columns:
                    # Set typical errors if not provided
                    if err_col not in cat.columns or np.all(cat[err_col] == 0):
                        cat[err_col] = np.where(
                            cat[mag_col] < 19,
                            0.1,  # Brighter stars
                            0.2   # Fainter stars
                        )

            return cat

        except Exception as e:
            raise ValueError(f"USNO-B query failed: {str(e)}")

    def _get_makak_data(self) -> Optional[astropy.table.Table]:
        """Get data from pre-filtered MAKAK catalog"""
        config = self.KNOWN_CATALOGS[self.MAKAK]

        # Read the pre-filtered catalog
        cat = astropy.table.Table.read(config['filepath'])

        # Convert coordinates to degrees if they have units
        if hasattr(cat['radeg'], 'unit'):
            cat['radeg'] = cat['radeg'].value
        if hasattr(cat['decdeg'], 'unit'):
            cat['decdeg'] = cat['decdeg'].value

        # Filter by field of view
        ctr = SkyCoord(self._query_params.ra*u.deg, self._query_params.dec*u.deg, frame='fk5')
        corner1 = SkyCoord((self._query_params.ra+self._query_params.width)*u.deg, (self._query_params.dec+self._query_params.height)*u.deg, frame='fk5')
        corner2 = SkyCoord((self._query_params.ra-self._query_params.width)*u.deg, (self._query_params.dec-self._query_params.height)*u.deg, frame='fk5')
        radius = max(corner1.separation(ctr), corner2.separation(ctr))
        logging.debug(f"catalog.makak: fov radius: {radius} {radius}")

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

        # Strip units from all columns to ensure compatibility
        for col in cat.colnames:
            if hasattr(cat[col], 'unit'):
                cat[col] = cat[col].value

        return cat

    @classmethod
    def from_file(cls: Type[TableType], filename: str) -> TableType:
        """Create catalog instance from a local file with proper metadata handling"""
        try:
            data = astropy.table.Table.read(filename)

            # Initialize catalog with data
            obj = cls(data.as_array())

            # Copy existing metadata
            obj.meta.update(data.meta)

            # Initialize catalog properties if not present
            if 'catalog_props' not in obj.meta:
                obj.meta['catalog_props'] = {
                    'catalog_name': 'local',
                    'description': f'Local catalog from {filename}',
                    'epoch': None,
                    'filters': {}
                }

            return obj

        except Exception as e:
            raise ValueError(f"Failed to read catalog from {filename}: {str(e)}")

    @property
    def description(self) -> str:
        """Get catalog description"""
        return self.meta.get('catalog_props', {}).get('description', "Unknown catalog")

    @property
    def filters(self) -> Dict[str, CatalogFilter]:
        """Get available filters"""
        filters_dict = self.meta.get('catalog_props', {}).get('filters', {})
        return {k: CatalogFilter(**v) for k, v in filters_dict.items()}

    @property
    def epoch(self) -> float:
        """Get catalog epoch"""
        return float(self.meta.get('catalog_props', {}).get('epoch'))

    def __array_finalize__(self, obj: Optional[astropy.table.Table]) -> None:
        """Ensure proper handling of metadata during numpy operations"""
        super().__array_finalize__(obj)
        if obj is None:
            return

        # Copy catalog properties if they exist
        if hasattr(obj, 'meta') and 'catalog_props' in obj.meta:
            if not hasattr(self, 'meta'):
                self.meta = {}
            self.meta['catalog_props'] = obj.meta['catalog_props'].copy()

    def copy(self, copy_data: bool = True) -> astropy.table.Table:
        """Create a copy ensuring catalog properties are preserved"""
        new_cat = super().copy(copy_data=copy_data)
        if 'catalog_props' in self.meta:
            new_cat.meta['catalog_props'] = self.meta['catalog_props'].copy()
        return new_cat

