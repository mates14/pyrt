#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
from contextlib import suppress

import numpy as np

import astropy
import astropy.io.fits
import astropy.wcs
import astropy.table
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation

# This is to silence a particular annoying warning (MJD not present in a fits file)
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

from sklearn.neighbors import KDTree

import zpnfit
import fotfit
from catalogs import get_atlas, get_catalog
from cat2det import remove_junk
from refit_astrometry import refit_astrometry
from file_utils import try_det, try_img, write_region_file
from config import parse_arguments
#from config import load_config

if sys.version_info[0]*1000+sys.version_info[1]<3008:
    print(f"Error: python3.8 or higher is required (this is python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]})")
    sys.exit(-1)

def airmass(z):
    """ Compute astronomical airmass according to Rozenberg(1966) """
    cz = np.cos(z)
    return 1/(cz + 0.025 * np.exp(-11*cz) )

def summag(magarray):
    """add two magnitudes of the same zeropoint"""
    return -2.5 * np.log10(np.sum( np.power(10.0, -0.4*np.array(magarray))) )

def check_filter(nearest_ind, det):
    ''' check filter and return the most probable filter '''

    bestflt="-"
    besterr=np.float64(9e99)
    bestzero=0

    filters = ['B', 'V', 'R', 'I', 'N', 'Sloan_r', 'Sloan_g', 'Sloan_i', 'Sloan_z']

    for flt in filters:
        x = []
        y = []
        dy = []

        for i, d in zip(nearest_ind, det):

            mag1 = np.float64(9e99)
            mag2 = np.float64(9e99)

            for k in i:
                mag1 = summag([mag1, cat[k][first_filter[flt]]])
                mag2 = summag([mag2, cat[k][second_filter[flt]]])

            if len(i)>0:

                try:
                    x = np.append(x, mag1 + color_termB[flt] * (mag2 - mag1))
                except:
                    x = np.append(x, mag1 + color_termB['R'] * (mag2 - mag1))
                y = np.append(y, np.float64(d['MAG_AUTO']))
                dy = np.append(dy, np.float64(d['MAGERR_AUTO']))

        if len(x)<0: print("No identified stars within image"); continue

        Zo = np.median(np.array(x)-np.array(y))
        Zoe = np.median(np.abs(np.array(x)-np.array(y)-Zo)) * 0.674490 #/np.sqrt(len(np.array(x)))

        logging.info("Filter: %s Zo: %f Zoe: %f"%(flt, Zo, Zoe))

        if besterr>Zoe:
            bestzero=Zo
            besterr=Zoe
            bestflt=flt

    return bestflt

def median_zeropoint(nearest_ind, det, cat, flt, forceAB=False):
    ''' check filter and return the most likely value '''

    # Bessel filter transforms by Lupton (2005)
    first_filter = \
        {     "B":"Sloan_g",       "V":"Sloan_g",       "R":"Sloan_r",       "I":'Sloan_i', \
              "N":"Sloan_r", "Sloan_r":"Sloan_r", "Sloan_g":"Sloan_g", "Sloan_i":"Sloan_i", \
        "Sloan_z":"Sloan_z", "Sloan_u":"Sloan_u",  "halpha":"Sloan_r",    "oiii":"Sloan_g", \
            "sii":"Sloan_i"} # basic filter to be fitted against
    second_filter = \
        {     "B":"Sloan_r",       "V":"Sloan_r",       "R":"Sloan_g",       "I":'Sloan_z', \
              "N":"Sloan_i", "Sloan_r":"Sloan_i", "Sloan_g":"Sloan_r", "Sloan_i":"Sloan_r", \
        "Sloan_z":"Sloan_i", "Sloan_u":"Sloan_g",  "halpha":"Sloan_i",    "oiii":"Sloan_r", \
            "sii":"Sloan_r"} # filter to compute color
    third_filter = \
        {     "B":"Sloan_i",       "V":"Sloan_i",       "R":"Sloan_i",       "I":'Sloan_r', \
              "N":"Sloan_g", "Sloan_r":"Sloan_g", "Sloan_g":"Sloan_i", "Sloan_i":"Sloan_z", \
        "Sloan_z":"Sloan_r", "Sloan_u":"Sloan_r",  "halpha":"Sloan_z",    "oiii":"Sloan_i", \
            "sii":"Sloan_z"} # filter to compute color
    color_term0 =  \
        {     "B":+0.2271,         "V":-0.0038,         "R":-0.0971,         "I":-0.3974,   \
              "N":0,         "Sloan_r":0,         "Sloan_g":0,         "Sloan_i":0,         \
        "Sloan_z":0,         "Sloan_u":0,          "halpha":0,            "oiii":0,         \
            "sii":0} # color term zeropoint
    color_termB = \
        {     "B":+0.3130,         "V":-0.5784,         "R":+0.1837,         "I":-0.3780,   \
              "N":0,         "Sloan_r":0,         "Sloan_g":0,         "Sloan_i":0,         \
        "Sloan_z":0,         "Sloan_u":0,          "halpha":0,            "oiii":0,         \
            "sii":0} # color index

#    bestflt="-"
#    besterr=np.float64(9e99)
#    bestzero=0

    x = []
    y = []
    dy = []

    for i, d in zip(nearest_ind, det):

        mag1 = np.float64(9e99)
        mag2 = np.float64(9e99)

#        try:
#            test=color_term0[flt]
#        except KeyError:
#            flt='R'

        if forceAB: zero=0
        else:
            try:
                zero=color_term0[flt]
            except KeyError:
                flt="V"
                zero=0

        for k in i:
            mag1 = summag([mag1, cat[k][first_filter[flt]]])
            mag2 = summag([mag2, cat[k][second_filter[flt]]])

        if len(i)>0:
            x = np.append(x, mag1 + color_termB[flt] * (mag2 - mag1))
            y = np.append(y, np.float64(d['MAG_AUTO']))
            dy = np.append(dy, np.float64(d['MAGERR_AUTO']))

    # median would complain but not crash if there are no identifications, this is a clean way out
    if len(x)<=0:
        # print("No identified stars within image")
        return np.nan, np.nan, 0

    Zo = np.median(np.array(x)-np.array(y))
    Zoe = np.median(np.abs(np.array(x)-np.array(y)-Zo)) * 0.674490 #/np.sqrt(len(np.array(x)))

    return Zo, Zoe, len(x)

def print_image_line(det, flt, Zo, Zoe, target=None, idnum=0):
    ''' print photometric status for an image '''
    if target==None:
        tarmag=Zo+det.meta['LIMFLX3']
        tarerr=">"
        tarstatus="not_found"
    else:
        tarmag=Zo+target['MAG_AUTO']
        tarerr="%6.3f"%(target['MAGERR_AUTO'])
        tarstatus="ok"

    print("%s %14.6f %14.6f %s %3.0f %6.3f %4d %7.3f %6.3f %7.3f %6.3f %7.3f %s %d %s %s"%(
        det.meta['FITSFILE'], det.meta['JD'], det.meta['JD']+det.meta['EXPTIME']/86400.0, flt, det.meta['EXPTIME'],
        det.meta['AIRMASS'], idnum, Zo, Zoe, (det.meta['LIMFLX10']+Zo),
        (det.meta['LIMFLX3']+Zo), tarmag, tarerr, 0, det.meta['OBSID'], tarstatus))

    return

def get_base_filter(det, options):
    '''set up or guess the base filter to use for fitting'''
    johnson_filters = ['Johnson_B', 'Johnson_V', 'Johnson_R', 'Johnson_I', 'B', 'V', 'R', 'I']
    fit_in_johnson = False
    basemag = 'Sloan_r'

    if options.johnson:
        basemag = 'Johnson_V'
        fit_in_johnson = True

    if options.guessbase:
        if det.meta['FILTER'] == 'Sloan_g': basemag = 'Sloan_g'
        if det.meta['FILTER'] == 'Sloan_r': basemag = 'Sloan_r'
        if det.meta['FILTER'] == 'Sloan_i': basemag = 'Sloan_i'
        if det.meta['FILTER'] == 'Sloan_z': basemag = 'Sloan_z'
        if det.meta['FILTER'] == 'g-SLOAN': basemag = 'Sloan_g'
        if det.meta['FILTER'] == 'r-SLOAN': basemag = 'Sloan_r'
        if det.meta['FILTER'] == 'i-SLOAN': basemag = 'Sloan_i'
        if det.meta['FILTER'] == 'z-SLOAN': basemag = 'Sloan_z'
        if det.meta['FILTER'] == 'Johnson_B': basemag = 'Johnson_B'
        if det.meta['FILTER'] == 'Johnson_V': basemag = 'Johnson_V'
        if det.meta['FILTER'] == 'Johnson_R': basemag = 'Johnson_R'
        if det.meta['FILTER'] == 'Johnson_I': basemag = 'Johnson_I'
        if det.meta['FILTER'] == 'g': basemag = 'Sloan_g'
        if det.meta['FILTER'] == 'r': basemag = 'Sloan_r'
        if det.meta['FILTER'] == 'i': basemag = 'Sloan_i'
        if det.meta['FILTER'] == 'z': basemag = 'Sloan_z'
        if det.meta['FILTER'] == 'B': basemag = 'Johnson_B'
        if det.meta['FILTER'] == 'V': basemag = 'Johnson_V'
        if det.meta['FILTER'] == 'R': basemag = 'Johnson_R'
        if det.meta['FILTER'] == 'I': basemag = 'Johnson_I'
        fit_in_johnson = bool(basemag in johnson_filters)

    if options.basemag is not None:
        fit_in_johnson = bool(options.basemag in johnson_filters)
        basemag = options.basemag

    logging.info(f"basemag={basemag} fit_in_johnson={fit_in_johnson}")
    return basemag, fit_in_johnson

class PhotometryData:
    def __init__(self):
        self._data = {}
        self._meta = {}
        self._masks = {}
        self._current_mask = None

    def init_column(self, name):
        """Initialize a new column."""
        if name not in self._data:
            self._data[name] = []

    def append(self, **kwargs):
        """Append data to multiple columns at once."""
        for name, value in kwargs.items():
            self.init_column(name)
            self._data[name].append(value)

    def extend(self, **kwargs):
        """Extend multiple columns with iterable data at once."""
        for name, value in kwargs.items():
            self.init_column(name)
            self._data[name].extend(value)

    def set_meta(self, key, value):
        """Set metadata."""
        self._meta[key] = value

    def get_meta(self, key, default=None):
        """Get metadata."""
        return self._meta.get(key, default)

    def finalize(self):
        """Convert list-based data to numpy arrays."""
        for name in self._data:
            self._data[name] = np.array(self._data[name])

        # Initialize a default mask that includes all data
        self.add_mask('default', np.ones(len(next(iter(self._data.values()))), dtype=bool))
        self.use_mask('default')

    def add_mask(self, name, mask):
        """Add a new mask or update an existing one."""
        self._masks[name] = mask

    def use_mask(self, name):
        """Set the current active mask."""
        if name not in self._masks:
            raise ValueError(f"Mask '{name}' does not exist.")
        self._current_mask = name

    def get_current_mask(self):
        """Get the current active mask."""
        if self._current_mask is None:
            raise ValueError("No mask is currently active.")
        return self._masks[self._current_mask]

    def get_arrays(self, *names):
        """Get multiple columns as separate numpy arrays, applying the current mask."""
        return tuple(self._data[name][self._masks[self._current_mask]] for name in names)

    def apply_mask(self, mask, name=None):
        """Apply a boolean mask to the current mask or create a new named mask."""
        if name is None:
            # Apply to current mask
            self._masks[self._current_mask] &= mask
        else:
            # Create a new mask
            self.add_mask(name, self._masks[self._current_mask] & mask)
            self.use_mask(name)

    def reset_mask(self, name=None):
        """Reset the specified mask or current mask to include all data."""
        if name is None:
            name = self._current_mask
        self._masks[name] = np.ones(len(next(iter(self._data.values()))), dtype=bool)

    def __len__(self):
        """Return the number of rows in the data (considering the current mask)."""
        if self._data:
            return np.sum(self._masks[self._current_mask])
        return 0

    def __repr__(self):
        """Return a string representation of the data."""
        return f"PhotometryData with columns: {list(self._data.keys())}, current mask: {self._current_mask}"

    def write(self, filename, format="ascii.ecsv", **kwargs):
        """Write the data to a file using astropy.table.Table.write method."""
        table = astropy.table.Table({k: v[self._masks[self._current_mask]] for k, v in self._data.items()})
        table.meta.update(self._meta)
        table.write(filename, format=format, **kwargs)

def make_pairs_to_fit(det, cat, nearest_ind, imgwcs, options, data):
    """
    Efficiently create pairs of data to be fitted.

    :param det: Detection table
    :param cat: Catalog table
    :param nearest_ind: Indices of nearest catalog stars for each detection
    :param imgwcs: WCS object for the image
    :param options: Command line options
    :param data: PhotometryData object to store results
    """
    try:
        # Create a mask for valid matches
        valid_matches = np.array([len(inds) > 0 for inds in nearest_ind])

        # Get all required data from det and cat
        det_data = np.array([det['X_IMAGE'], det['Y_IMAGE'], det['MAG_AUTO'], det['MAGERR_AUTO'], det['ERRX2_IMAGE'], det['ERRY2_IMAGE']]).T[valid_matches]
        cat_inds = np.array([inds[0] if len(inds) > 0 else -1 for inds in nearest_ind])[valid_matches]

        # Extract relevant catalog data
        cat_data = cat[cat_inds]

        # Compute celestial coordinates
        ra, dec = imgwcs.all_pix2world(det_data[:, 0], det_data[:, 1], 1)

        # Compute airmass
        loc = EarthLocation(lat=det.meta['LATITUDE']*u.deg,
                            lon=det.meta['LONGITUD']*u.deg,
                            height=det.meta['ALTITUDE']*u.m)
        time = Time(det.meta['JD'], format='jd')
        coords = SkyCoord(ra*u.deg, dec*u.deg)
        altaz = coords.transform_to(AltAz(obstime=time, location=loc))
        airmass = altaz.secz.value

        # Compute normalized coordinates
        coord_x = (det_data[:, 0] - det.meta['CTRX']) / 1024
        coord_y = (det_data[:, 1] - det.meta['CTRY']) / 1024

        # Apply magnitude and color limits
        if det.meta['REJC']:
            mag0, mag1, mag2, mag3, mag4 = [cat_data[f] for f in ['Johnson_B', 'Johnson_V', 'Johnson_R', 'Johnson_I', 'J']]
        else:
            mag0, mag1, mag2, mag3, mag4 = [cat_data[f] for f in ['Sloan_g', 'Sloan_r', 'Sloan_i', 'Sloan_z', 'J']]

        magcat = cat_data[det.meta['REFILTER']]

        # Create masks for magnitude and color limits
        mag_mask = (magcat >= options.brightlim) & (magcat <= options.maglim) if options.brightlim else (magcat <= options.maglim)
        color_mask = ((mag0 - mag2)/2 <= options.redlim) & ((mag0 - mag2)/2 >= options.bluelim) if options.redlim and options.bluelim else True

        final_mask = mag_mask & color_mask

        # Calculate errors
        temp_dy = det_data[:, 3][final_mask]
        temp_dy_no_zero = np.sqrt(np.power(temp_dy,2)+0.0004)

        _dx = det_data[:, 4][final_mask]
        _dy = det_data[:, 5][final_mask]
        _image_dxy = np.sqrt(np.power(_dx,2) + np.power(_dy,2) + 0.0025)  # Do not trust errors better than 1/20 pixel

        # Update PhotometryData object
        data.extend(
            x=magcat[final_mask],
            aabs=airmass[final_mask],
            image_x = det_data[:, 0][final_mask],
            image_y = det_data[:, 1][final_mask],
            color1=(mag0 - mag1)[final_mask],
            color2=(mag1 - mag2)[final_mask],
            color3=(mag2 - mag3)[final_mask],
            color4=(mag3 - mag4)[final_mask],
            y=det_data[:, 2][final_mask],
            dy=temp_dy_no_zero,
            ra=cat_data['radeg'][final_mask],
            dec=cat_data['decdeg'][final_mask],
            adif=airmass[final_mask] - det.meta['AIRMASS'],
            coord_x=coord_x[final_mask],
            coord_y=coord_y[final_mask],
            image_dxy = _image_dxy,
            img=np.full(np.sum(final_mask), det.meta['IMGNO'])
        )
    except KeyError as e:
        print(f"Error: Missing key in detection or catalog data: {e}")
    except ValueError as e:
        print(f"Error: Invalid value encountered: {e}")
    except Exception as e:
        print(f"Unexpected error in make_pairs_to_fit: {e}")

def write_stars_file(data, ffit, imgwcs, filename="stars"):
    """
    Write star data to a file for visualization purposes.

    Parameters:
    data : PhotometryData
        The PhotometryData object containing all star data
    ffit : object
        The fitting object containing the model and fit results
    filename : str, optional
        The name of the output file (default is "stars")
    """
    # Ensure we're using the most recent mask (likely the combined photometry and astrometry mask)
    #current_mask = data.get_current_mask()
    current_mask = data.get_current_mask()
    data.use_mask('default')

    # Get all required arrays
    x, y, adif, coord_x, coord_y, color1, color2, color3, color4, img, dy, ra, dec, image_x, image_y = data.get_arrays(
        'x', 'y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'dy', 'ra', 'dec', 'image_x', 'image_y'
    )

    # Calculate model magnitudes
    model_input = (y, adif, coord_x, coord_y, color1, color2, color3, color4, img, x, dy)
    model_mags = ffit.model(np.array(ffit.fitvalues), model_input)

    # Calculate astrometric residuals (if available)
    try:
        astx, asty = imgwcs.all_world2pix( ra, dec, 1)
        ast_residuals = np.sqrt((astx - coord_x)**2 + (asty - coord_y)**2)
    except KeyError:
        ast_residuals = np.zeros_like(x)  # If astrometric data is not available

    # Create a table with all the data
    stars_table = astropy.table.Table([
        x, adif, image_x, image_y, color1, color2, color3, color4,
        model_mags, dy, ra, dec, astx, asty, ast_residuals, current_mask, current_mask, current_mask
    ], names=[
        'cat_mags', 'airmass', 'image_x', 'image_y', 'color1', 'color2', 'color3', 'color4',
        'model_mags', 'mag_err', 'ra', 'dec', 'ast_x', 'ast_y', 'ast_residual', 'mask', 'mask2', 'mask3'
    ])

    # Add column descriptions
    stars_table['cat_mags'].description = 'Catalog magnitude'
    stars_table['airmass'].description = 'Airmass difference from mean'
    stars_table['image_x'].description = 'X coordinate in image'
    stars_table['image_y'].description = 'Y coordinate in image'
    stars_table['color1'].description = 'Color index 1'
    stars_table['color2'].description = 'Color index 2'
    stars_table['color3'].description = 'Color index 3'
    stars_table['color4'].description = 'Color index 4'
    stars_table['model_mags'].description = 'Observed magnitude'
    stars_table['mag_err'].description = 'Magnitude error'
    stars_table['ra'].description = 'Right Ascension'
    stars_table['dec'].description = 'Declination'
    stars_table['ast_x'].description = 'Model position X'
    stars_table['ast_y'].description = 'Model position Y'
    stars_table['ast_residual'].description = 'Astrometric residual'
    stars_table['mask'].description = 'Boolean mask (True for included points)'
    stars_table['mask2'].description = 'Boolean mask (True for included points)'
    stars_table['mask3'].description = 'Boolean mask (True for included points)'

    # Write the table to a file
    stars_table.write(filename, format='ascii.ecsv', overwrite=True)

def perform_photometric_fitting(data, options, zeropoints):
    """
    Perform photometric fitting on the data.

    Args:
    data (PhotometryData): Object containing all photometry data.
    options (argparse.Namespace): Command line options.
    zeropoints (list): Initial zeropoints for each image.

    Returns:
    fotfit.fotfit: The fitted photometry model.
    """
    ffit = fotfit.fotfit(fit_xy=options.fit_xy)

    # Set up the model
    setup_photometric_model(ffit, options)

    # Set initial zeropoints
    ffit.zero = zeropoints

    # Perform initial fit
    fdata = data.get_arrays('y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'x', 'dy')
    ffit.fit(fdata)

    # Apply photometric mask and refit
    photo_mask = ffit.residuals(ffit.fitvalues, fdata) < 5 * ffit.wssrndf
    data.apply_mask(photo_mask, 'photometry')
    data.use_mask('photometry')
    fdata = data.get_arrays('y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'x', 'dy')

    ffit.delin = True
    ffit.fit(fdata)

    # Final fit with refined mask
    data.use_mask('default')
    fdata = data.get_arrays('y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'x', 'dy')
    photo_mask = ffit.residuals(ffit.fitvalues, fdata) < 5 * ffit.wssrndf
    data.apply_mask(photo_mask, 'photometry')
    data.use_mask('photometry')
    fdata = data.get_arrays('y', 'adif', 'coord_x', 'coord_y', 'color1', 'color2', 'color3', 'color4', 'img', 'x', 'dy')
    ffit.fit(fdata)

    print(f"Final fit variance: {ffit.wssrndf}")

    return ffit

def setup_photometric_model(ffit, options):
    """
    Set up the photometric model based on command line options.

    Args:
    ffit (fotfit.fotfit): The photometric fitting object.
    options (argparse.Namespace): Command line options.
    """
    # Load model from file if specified
    if options.model:
        load_model_from_file(ffit, options.model)

    ffit.fixall()  # Fix all terms initially
    ffit.fixterm(["N1"], values=[0])

    # Set up fitting terms
    if options.terms:
        setup_fitting_terms(ffit, options.terms, options.verbose, options.fit_xy)

def load_model_from_file(ffit, model_file):
    """
    Load a photometric model from a file.

    Args:
    ffit (fotfit.fotfit): The photometric fitting object.
    model_file (str): Path to the model file.
    """
    model_paths = [
        model_file,
        f"/home/mates/pyrt/model/{model_file}.mod",
        f"/home/mates/pyrt/model/{model_file}-{ffit.det.meta['FILTER']}.mod"
    ]
    for path in model_paths:
        try:
            ffit.readmodel(path)
            print(f"Model imported from {path}")
            return
        except:
            pass
    print(f"Cannot open model {model_file}")

def setup_fitting_terms(ffit, terms, verbose, fit_xy):
    """
    Set up fitting terms based on the provided options.

    Args:
    ffit (fotfit.fotfit): The photometric fitting object.
    terms (str): Comma-separated list of terms to fit.
    verbose (bool): Whether to print verbose output.
    fit_xy (bool): Whether to fit xy tilt for each image separately.
    """
    for term in terms.split(","):
        if term == "":
            continue  # Skip empty terms
        if term[0] == '.':
            setup_polynomial_terms(ffit, term, verbose, fit_xy)
        else:
            ffit.fitterm([term], values=[1e-6])

def setup_polynomial_terms(ffit, term, verbose, fit_xy):
    """
    Set up polynomial terms for fitting.

    Args:
    ffit (fotfit.fotfit): The photometric fitting object.
    term (str): The polynomial term descriptor.
    verbose (bool): Whether to print verbose output.
    fit_xy (bool): Whether to fit xy tilt for each image separately.
    """
    if term[1] == 'p':
        setup_surface_polynomial(ffit, term, verbose, fit_xy)
    elif term[1] == 'r':
        setup_radial_polynomial(ffit, term, verbose)

def setup_surface_polynomial(ffit, term, verbose, fit_xy):
    """Set up surface polynomial terms."""
    pol_order = int(term[2:])
    if verbose:
        print(f"Setting up a surface polynomial of order {pol_order}:")
    polytxt = "P(x,y)="
    for pp in range(1, pol_order + 1):
        if fit_xy and pp == 1:
            continue
        for rr in range(0, pp + 1):
            polytxt += f"+P{rr}X{pp-rr}Y*x**{rr}*y**{pp-rr}"
            ffit.fitterm([f"P{rr}X{pp-rr}Y"], values=[1e-6])
    if verbose:
        print(polytxt)

def setup_radial_polynomial(ffit, term, verbose):
    """Set up radial polynomial terms."""
    pol_order = int(term[2:])
    if verbose:
        print(f"Setting up a polynomial of order {pol_order} in radius:")
    polytxt = "P(r)="
    for pp in range(1, pol_order + 1):
        polytxt += f"+P{pp}R*(x**2+y**2)**{pp}"
        ffit.fitterm([f"P{pp}R"], values=[1e-6])
    if verbose:
        print(polytxt)

def update_det_file(fitsimgf: str, options): #  -> Tuple[Optional[astropy.table.Table], str]:
    """Update the .det file using cat2det.py."""
    cmd = ["cat2det.py"]
    if options.verbose:
        cmd.append("-v")
    if options.filter:
        cmd.extend(["-f", options.filter])
    cmd.append(fitsimgf)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running cat2det.py: {e}")
        return None, ""
    
    return try_det(os.path.splitext(fitsimgf)[0] + ".det", options.verbose)

def should_update_det_file(det, fitsimg, detf, fitsimgf, options):
    """Determine if the .det file should be updated."""
    if det is None and fitsimg is not None:
        return True
    if det is not None and fitsimg is not None and options.autoupdate:
        return os.path.getmtime(fitsimgf) > os.path.getmtime(detf)
    return False

def process_input_file(arg, options):

    detf, det = try_det(arg, options.verbose)
    if det is None: detf, det = try_det(os.path.splitext(arg)[0] + ".det", options.verbose)
    if det is None: detf, det = try_det(arg + ".det", options.verbose)

    fitsimgf, fitsimg = try_img(arg + ".fits", options.verbose)
    if fitsimg is None: fitsimgf, fitsimg = try_img(os.path.splitext(arg)[0] + ".fits", options.verbose)
    if fitsimg is None: fitsimgf, fitsimg = try_img(os.path.splitext(arg)[0], options.verbose)

    # with these, we should have the possible filenames covered, now lets see what we got
    # 1. have fitsimg, no .det: call cat2det, goto 3
    # 2. have fitsimg, have det, det is older than fitsimg: same as 1
    if should_update_det_file(det, fitsimg, detf, fitsimgf, options):
        detf, det = update_det_file(fitsimgf, options)
        logging.info("Back in dophot")

    if det is None: return None
    # print(type(det.meta)) # OrderedDict, so we can:
    with suppress(KeyError): det.meta.pop('comments')
    with suppress(KeyError): det.meta.pop('history')

    # 3. have fitsimg, have .det, note the fitsimg filename, close fitsimg and run
    # 4. have det, no fitsimg: just run, writing results into fits will be disabled
    if fitsimgf is not None: det.meta['filename'] = fitsimgf
    det.meta['detf'] = detf
    logging.info(f"DetF={detf}, ImgF={fitsimgf}")

    return det

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def write_results(data, ffit, options, alldet, target):
    zero, zerr = ffit.zero_val()

    for img, det in enumerate(alldet):
        start = time.time()
        if options.astrometry:
            try:
                astropy.io.fits.setval(os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "LIMMAG", 0, value=zero[img]+det.meta['LIMFLX3'])
                astropy.io.fits.setval(os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "MAGZERO", 0, value=zero[img])
                astropy.io.fits.setval(os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "RESPONSE", 0, value=ffit.oneline())
            except Exception as e:
                logging.warning(f"Writing LIMMAG/MAGZERO/RESPONSE to an astrometrized image failed: {e}")
        logging.info(f"Writing to astrometrized image took {time.time()-start:.3f}s")

        start = time.time()
        fn = os.path.splitext(det.meta['detf'])[0] + ".ecsv"
        det['MAG_CALIB'] = ffit.model(np.array(ffit.fitvalues), (det['MAG_AUTO'], det.meta['AIRMASS'],
            det['X_IMAGE']/1024-det.meta['CTRX']/1024, det['Y_IMAGE']/1024-det.meta['CTRY']/1024,0,0,0,0, img,0,0))
        det['MAGERR_CALIB'] = np.sqrt(np.power(det['MAGERR_AUTO'],2)+np.power(zerr[img],2))

        det.meta['MAGZERO'] = zero[img]
        det.meta['DMAGZERO'] = zerr[img]
        det.meta['MAGLIMIT'] = det.meta['LIMFLX3']+zero[img]
        det.meta['WSSRNDF'] = ffit.wssrndf
        det.meta['RESPONSE'] = ffit.oneline()

        if (zerr[img] < 0.2) or (options.reject is None):
            det.write(fn, format="ascii.ecsv", overwrite=True)
        else:
            if options.verbose:
                logging.warning("rejected (too large uncertainty in zeropoint)")
            with open("rejected", "a+") as out_file:
                out_file.write(f"{det.meta['FITSFILE']} {ffit.wssrndf:.6f} {zerr[img]:.3f}\n")
            sys.exit(0)
        logging.info(f"Saving ECSV output took {time.time()-start:.3f}s")
        
        if options.flat or options.weight: 
            start = time.time()
            ffData = mesh_create_flat_field_data(det, ffit)
            logging.info(f"Generating the flat field took {time.time()-start:.3f}s")
        if options.weight: save_weight_image(det, ffData, options)
        if options.flat: save_flat_image(det, ffData)

        write_output_line(det, options, zero[img], zerr[img], ffit, target[img])

def mesh_create_flat_field_data(det, ffit):
    """
    Create the flat field data array.
    
    :param det: Detection table containing metadata
    :param ffit: Fitting object containing the flat field model
    :return: numpy array of flat field data
    """
    logging.info("Generating the flat-field array")
    shape = (det.meta['IMGAXIS2'], det.meta['IMGAXIS1'])
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    
    return ffit.mesh_flat(x, y, 
                     ctrx=det.meta['CTRX'], 
                     ctry=det.meta['CTRY'], 
                     img=det.meta['IMGNO'])

def old_create_flat_field_data(det, ffit):
    """
    Create the flat field data array.
    
    :param det: Detection table containing metadata
    :param ffit: Fitting object containing the flat field model
    :return: numpy array of flat field data
    """
    logging.info("Generating the flat-field array")
    return np.fromfunction(ffit.old_flat, [det.meta['IMGAXIS2'], det.meta['IMGAXIS1']],
        ctrx=det.meta['CTRX'], ctry=det.meta['CTRY'], img=det.meta['IMGNO'])

def save_weight_image(det, ffData, options):
    """
    Generate and save the weight image.
    
    :param det: Detection table containing metadata
    :param ffData: Flat field data array
    :param options: Command line options
    """
    start = time.time()
    gain = options.gain if options.gain is not None else 2.3
    wwData = np.power(10, -0.4 * (22 - ffData))
    wwData = np.power(wwData * gain, 2) / (wwData * gain + 3.1315 * np.power(det.meta['BGSIGMA'] * gain * det.meta['FWHM'] / 2, 2))
    wwHDU = astropy.io.fits.PrimaryHDU(data=wwData)
    weightfile = os.path.splitext(det.meta['filename'])[0] + "w.fits"

    if os.path.isfile(weightfile):
        logging.warning(f"Operation will overwrite an existing image: {weightfile}")
        os.unlink(weightfile)

    with astropy.io.fits.open(weightfile, mode='append') as wwFile:
        wwFile.append(wwHDU)
    
    logging.info(f"Weight image saved to {weightfile}")
    logging.info(f"Writing the weight file took {time.time()-start:.3f}s")

def save_flat_image(det, ffData):
    """
    Generate and save the flat image.
    
    :param det: Detection table containing metadata
    :param ffData: Flat field data array
    """
    start = time.time()
    flatData = np.power(10, 0.4 * (ffData - 8.9))
    ffHDU = astropy.io.fits.PrimaryHDU(data=flatData)
    flatfile = os.path.splitext(det.meta['filename'])[0] + "f.fits"

    if os.path.isfile(flatfile):
        logging.warning(f"Operation will overwrite an existing image: {flatfile}")
        os.unlink(flatfile)

    with astropy.io.fits.open(flatfile, mode='append') as ffFile:
        ffFile.append(ffHDU)
    
    logging.info(f"Flat image saved to {flatfile}")
    logging.info(f"Writing the flatfield file took {time.time()-start:.3f}s")

def write_output_line(det, options, zero, zerr, ffit, target):
    tarid = det.meta.get('TARGET', 0)
    obsid = det.meta.get('OBSID', 0)

    chartime = det.meta['JD'] + det.meta['EXPTIME'] / 2
    if options.date == 'char':
        chartime = det.meta.get('CHARTIME', chartime)
    elif options.date == 'bjd':
        chartime = det.meta.get('BJD', chartime)

    if target is not None:
        tarx = target['X_IMAGE']/1024 - det.meta['CTRX']/1024
        tary = target['Y_IMAGE']/1024 - det.meta['CTRY']/1024
        data_target = np.array([[target['MAG_AUTO']], [det.meta['AIRMASS']], [tarx], [tary], [0], [0], [0], [0], [det.meta['IMGNO']], [0], [0]])
        mo = ffit.model(np.array(ffit.fitvalues), data_target)

        out_line = f"{det.meta['FITSFILE']} {det.meta['JD']:.6f} {chartime:.6f} {det.meta['FILTER']} {det.meta['EXPTIME']:3.0f} {det.meta['AIRMASS']:6.3f} {det.meta['IDNUM']:4d} {zero:7.3f} {zerr:6.3f} {det.meta['LIMFLX3']+zero:7.3f} {ffit.wssrndf:6.3f} {mo[0]:7.3f} {target['MAGERR_AUTO']:6.3f} {tarid} {obsid} ok"
    else:
        out_line = f"{det.meta['FITSFILE']} {det.meta['JD']:.6f} {chartime:.6f} {det.meta['FILTER']} {det.meta['EXPTIME']:3.0f} {det.meta['AIRMASS']:6.3f} {det.meta['IDNUM']:4d} {zero:7.3f} {zerr:6.3f} {det.meta['LIMFLX3']+zero:7.3f} {ffit.wssrndf:6.3f}  99.999  99.999 {tarid} {obsid} not_found"

    print(out_line)

    try:
        with open("dophot.dat", "a") as out_file:
            out_file.write(out_line + "\n")
    except Exception as e:
        print(f"Error writing to dophot.dat: {e}")

# ******** main() **********

def main():
    '''Take over the world.'''
    options = parse_arguments()
    setup_logging(options.verbose)

    logging.info(f"{os.path.basename(sys.argv[0])} running in Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")
    logging.info(f"Magnitude limit set to {options.maglim}")

    data = PhotometryData()

    target = []
    zeropoints = []
    metadata = []
    alldet=[]
    imgno=0

    for arg in options.files:

        det = process_input_file(arg, options)
        if det is None:
            logging.warning(f"I do not know what to do with {arg}")
            continue

#    some_file = open("det.reg", "w+")
#    some_file.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=3"\
#        " font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
#    for w_ra, w_dec in zip(det['X_IMAGE'],det['Y_IMAGE']):
#        some_file.write(f"circle({w_ra:.7f},{w_dec:.7f},3\") # color=red width=3\n")
#    some_file.close()

        imgwcs = astropy.wcs.WCS(det.meta)

        det['ALPHA_J2000'], det['DELTA_J2000'] = imgwcs.all_pix2world( [det['X_IMAGE']], [det['Y_IMAGE']], 1)

        # if options.usewcs is not None:
        #      fixme from "broken_usewcs.py"            

        if options.enlarge is not None:
            enlarge = options.enlarge
        else: enlarge=1

        start = time.time()
        if options.makak:
            cat = astropy.table.Table.read('/home/mates/test/catalog.fits')
            ctr = SkyCoord(det.meta['CTRRA']*astropy.units.deg,det.meta['CTRDEC']*astropy.units.deg,frame='fk5')
            c2 = SkyCoord(cat['radeg']*astropy.units.deg,cat['decdeg']*astropy.units.deg,frame='fk5').separation(ctr) < \
                det.meta['FIELD']*astropy.units.deg / 2
            cat=cat[c2]
        else:
            if options.catalog:
                cat = get_catalog(options.catalog, mlim=options.maglim)
            else:
                cat = get_atlas(det.meta['CTRRA'], det.meta['CTRDEC'], width=enlarge*det.meta['FIELD'],
                    height=enlarge*det.meta['FIELD'], mlim=options.maglim)

        # Epoch for PM correction (Gaia DR2@2015.5)
        #epoch = ( det.meta['JD'] - 2457204.5 ) / 365.2425 # Epoch for PM correction (Gaia DR2@2015.5)
        epoch = ( (det.meta['JD'] - 2440587.5) - (cat.meta['astepoch'] - 1970.0)*365.2425  ) / 365.2425
        logging.info(f"EPOCH_DIFF={epoch}")
        cat['radeg'] += epoch*cat['pmra']
        cat['decdeg'] += epoch*cat['pmdec']

        logging.info(f"Catalog search took {time.time()-start:.3f}s")
        logging.info(f"Catalog contains {len(cat)} entries")

        # write_region_file(cat, "cat.reg")

        if options.idlimit: idlimit = options.idlimit
        else:
            try:
                idlimit = det.meta['FWHM']
                logging.info(f"idlimit set to fits header FWHM value of {idlimit} pixels.")
            except KeyError:
                idlimit = 2.0/3600
                logging.info(f"idlimit set to a hard-coded default of {idlimit} pixels.")

        # ===  identification with KDTree  ===
        Y = np.array([det['X_IMAGE'], det['Y_IMAGE']]).transpose()

        start = time.time()
        try:
            Xt = np.array(imgwcs.all_world2pix(cat['radeg'], cat['decdeg'],1))
        except (ValueError,KeyError):
            logging.warning(f"Astrometry of {arg} sucks! Skipping.")
            continue

        # clean up the array from off-image objects
        cat = cat[np.logical_not(np.any([
            np.isnan(Xt[0]),
            np.isnan(Xt[1]),
            Xt[0]<0,
            Xt[1]<0,
            Xt[0]>det.meta['IMGAXIS1'],
            Xt[1]>det.meta['IMGAXIS2']
            ], axis=0))]

        Xt = np.array(imgwcs.all_world2pix(cat['radeg'], cat['decdeg'],1))
        X = Xt.transpose()

        tree = KDTree(X)
        nearest_ind, nearest_dist = tree.query_radius(Y, r=idlimit, return_distance=True, count_only=False)
        logging.info(f"Object cross-id took {time.time()-start:.3f}s")

        # identify the target object
        target.append(None)
        if det.meta['OBJRA']<-99 or det.meta['OBJDEC']<-99:
            logging.info("Target was not defined")
        else:
            logging.info(f"Target coordinates: {det.meta['OBJRA']:.6f} {det.meta['OBJDEC']:+6f}")
            Z = np.array(imgwcs.all_world2pix([det.meta['OBJRA']],[det.meta['OBJDEC']],1)).transpose()

            if np.isnan(Z[0][0]) or np.isnan(Z[0][1]):
                object_ind = None
                logging.warning("Target transforms to Nan... :(")
            else:
                treeZ = KDTree(Z)
                object_ind, object_dist = treeZ.query_radius(Y, r=idlimit, return_distance=True, count_only=False)

                if object_ind is not None:
                    mindist = 2*idlimit # this is just a big number
                    for i, dd, d in zip(object_ind, object_dist, det):
                        if len(i) > 0 and dd[0] < mindist:
                            target[imgno] = d
                            mindist = dd[0]
#                        if target[imgno] is None:
#                            logging.info("Target was not found")
#                        else:
#                            logging.info(f"Target is object id {target[imgno]['NUMBER']} at distance {mindist:.2f} px")

        # === objects identified ===

        # TODO: handle discrepancy between estimated and fitsheader filter
        flt = det.meta['FILTER']

        if options.tryflt:
            fltx = check_filter(nearest_ind, det)
    #        print("Filter: %s"%(fltx))
            flt = fltx

        # crude estimation of an image photometric zeropoint
        Zo, Zoe, Zn = median_zeropoint(nearest_ind, det, cat, flt)

        if Zn == 0:
            print(f"Warning: no identified stars in {det.meta['FITSFILE']}, skip image")
            continue

        if options.tryflt:
            print_image_line(det, flt, Zo, Zoe, target[imgno], Zn)
            sys.exit(0)

        # === fill up fields to be fitted ===
        zeropoints=zeropoints+[Zo]
        det.meta['IDNUM'] = Zn
        metadata.append(det.meta)

        cas = astropy.time.Time(det.meta['JD'], format='jd')
        loc = astropy.coordinates.EarthLocation(\
            lat=det.meta['LATITUDE']*astropy.units.deg, \
            lon=det.meta['LONGITUD']*astropy.units.deg, \
            height=det.meta['ALTITUDE']*astropy.units.m)

        tmpra, tmpdec = imgwcs.all_pix2world(det.meta['CTRX'], det.meta['CTRY'], 1)
        rd = astropy.coordinates.SkyCoord( np.float64(tmpra), np.float64(tmpdec), unit=astropy.units.deg)
        rdalt = rd.transform_to(astropy.coordinates.AltAz(obstime=cas, location=loc))
        det.meta['AIRMASS'] = airmass(np.pi/2-rdalt.alt.rad)

        det.meta['REFILTER'], det.meta['REJC'] = get_base_filter(det, options)

        # make pairs to be fitted
        det.meta['IMGNO'] = imgno
        make_pairs_to_fit(det, cat, nearest_ind, imgwcs, options, data)

        #if len(data)==0: # if number of stars from this file is 0, do not increment imgno, effectively skipping the file
        #    print("Not enough stars to work with (Records:", len(data),")")
        #    sys.exit(0)

        alldet = alldet+[det]
        imgno=imgno+1

    data.finalize()

    if imgno == 0:
        print("No images found to work with, quit")
        sys.exit(0)

    if len(data) == 0:
        print("No objects found to work with, quit")
        sys.exit(0)

    logging.info(f"Photometry will be fitted with {len(data)} objects from {imgno} files")
    # tady mame hvezdy ze snimku a hvezdy z katalogu nactene a identifikovane

    # Usage in main function:
    start = time.time()
    ffit = perform_photometric_fitting(data, options, zeropoints)
    logging.info(ffit)
    logging.info(f"Photometric fit took {time.time()-start:.3f}s")

    if options.reject:
        if ffit.wssrndf > options.reject:
            logging.info("rejected (too large reduced chi2)")
            out_file = open("rejected", "a+")
            out_file.write("%s %.6f -\n"%(metadata[0]['FITSFILE'], ffit.wssrndf))
            out_file.close()
            sys.exit(0)

    if options.save_model is not None:
        ffit.savemodel(options.save_model)

    # """ REFIT ASTROMETRY """
    if options.astrometry:
        start = time.time()
        zpntest = refit_astrometry(det, data, options)
        logging.info(f"Astrometric fit took {time.time()-start:.3f}s")
        if zpntest is not None:
            # Update FITS file with new WCS
            start = time.time()
            fitsbase = os.path.splitext(arg)[0]
            newfits = fitsbase + "t.fits"
            if os.path.isfile(newfits):
                logging.info(f"Will overwrite {newfits}")
                os.unlink(newfits)
            os.system(f"cp {fitsbase}.fits {newfits}")
            zpntest.write(newfits)
            imgwcs = astropy.wcs.WCS(zpntest.wcs())
            logging.info(f"Saving a new fits with WCS took {time.time()-start:.3f}s")
    # ASTROMETRY END

    if options.stars:
        start = time.time()
        write_stars_file(data, ffit, imgwcs)
        logging.info(f"Saving the stars file took {time.time()-start:.3f}s")

    write_results(data, ffit, options, alldet, target)

# this way, the variables local to main() are not globally available, avoiding some programming errors
if __name__ == "__main__":
    main()
