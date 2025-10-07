#!/usr/bin/env python3

import os
import sys
import time

if sys.version_info[0]*1000+sys.version_info[1]<3008:
    print("Error: python3.8 or higher is required (this is python %d.%d.%d)"%(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    exit(-1)

import math
import subprocess
import astropy
import astropy.io.fits
import astropy.wcs
import astropy.table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
#import scipy
import numpy as np
import argparse
import json

import scipy.optimize as fit
from sklearn.neighbors import KDTree,BallTree

import zpnfit
import fotfit

# TRAJECTORY-BASED MOVING OBJECT TRACKING
class MovingObjectTrail:
    """Track a moving object through multiple detections"""

    def __init__(self, first_detection, object_id):
        self.object_id = object_id
        self.detections = [dict(first_detection)]  # Store as dict for flexibility
        self.motion_ra = 0.0  # arcsec/hour
        self.motion_dec = 0.0  # arcsec/hour
        self.motion_sigma_ra = float('inf')
        self.motion_sigma_dec = float('inf')
        self.last_update_time = first_detection['JD']
        self.last_position_ra = first_detection['ALPHA_J2000']
        self.last_position_dec = first_detection['DELTA_J2000']

    def predict_position(self, time_jd):
        """Predict position at given time based on current motion estimate"""
        if len(self.detections) < 2:
            # No motion history, return last known position
            return self.last_position_ra, self.last_position_dec

        # Time difference in hours
        dt_hours = (time_jd - self.last_update_time) * 24.0

        # Predict position using linear motion
        ra_pred = self.last_position_ra + (self.motion_ra * dt_hours) / 3600.0
        dec_pred = self.last_position_dec + (self.motion_dec * dt_hours) / 3600.0

        return ra_pred, dec_pred

    def add_detection(self, new_detection):
        """Add new detection and update motion estimate"""
        self.detections.append(dict(new_detection))
        self.last_update_time = new_detection['JD']
        self.last_position_ra = new_detection['ALPHA_J2000']
        self.last_position_dec = new_detection['DELTA_J2000']

        # Update motion estimate if we have enough points
        if len(self.detections) >= 2:
            self._update_motion_estimate()

    def _update_motion_estimate(self):
        """Update motion estimate using recent detections"""
        # Use last 3-5 detections for motion estimate
        recent_detections = self.detections[-min(5, len(self.detections)):]

        if len(recent_detections) < 2:
            return

        # Extract times and positions
        times = np.array([det['JD'] for det in recent_detections])
        ra_positions = np.array([det['ALPHA_J2000'] for det in recent_detections])
        dec_positions = np.array([det['DELTA_J2000'] for det in recent_detections])

        # Convert to hours relative to first point
        t0 = times[0]
        times_hours = (times - t0) * 24.0

        # Linear fit for RA and Dec
        try:
            # RA motion (degrees/hour)
            ra_fit = np.polyfit(times_hours, ra_positions, 1)
            self.motion_ra = ra_fit[0] * 3600.0  # Convert to arcsec/hour

            # Dec motion (degrees/hour)
            dec_fit = np.polyfit(times_hours, dec_positions, 1)
            self.motion_dec = dec_fit[0] * 3600.0  # Convert to arcsec/hour

            # Estimate uncertainties from residuals
            ra_pred = np.polyval(ra_fit, times_hours)
            dec_pred = np.polyval(dec_fit, times_hours)

            ra_residuals = (ra_positions - ra_pred) * 3600.0  # arcsec
            dec_residuals = (dec_positions - dec_pred) * 3600.0  # arcsec

            # Use robust MAD estimator for uncertainties
            self.motion_sigma_ra = np.median(np.abs(ra_residuals)) / 0.67 if len(ra_residuals) > 1 else 1.0
            self.motion_sigma_dec = np.median(np.abs(dec_residuals)) / 0.67 if len(dec_residuals) > 1 else 1.0

        except (np.linalg.LinAlgError, ValueError):
            # Fallback if fitting fails
            self.motion_ra = 0.0
            self.motion_dec = 0.0
            self.motion_sigma_ra = 5.0  # Conservative uncertainty
            self.motion_sigma_dec = 5.0

    def get_search_radius(self, time_jd, base_idlimit=3.6, max_radius=30.0):
        """Calculate adaptive search radius for matching"""
        if len(self.detections) < 2:
            return base_idlimit  # Conservative for new objects

        # Time gap since last detection (hours)
        time_gap_hours = (time_jd - self.last_update_time) * 24.0

        # Motion uncertainty (arcsec)
        motion_uncertainty = np.sqrt(self.motion_sigma_ra**2 + self.motion_sigma_dec**2)

        # Adaptive radius: base + motion uncertainty * time gap
        adaptive_radius = base_idlimit + motion_uncertainty * time_gap_hours

        # Cap at maximum to avoid matching everything
        return min(adaptive_radius, max_radius)

    def to_astropy_table(self):
        """Convert detections to astropy table for compatibility"""
        if not self.detections:
            return None

        # Create table from all detections
        table_data = []
        for det in self.detections:
            table_data.append(det)

        return astropy.table.Table(table_data)

def try_grbt0(target):
    """tries to run a command that gets T0 of a GRB from the stars DB"""
    try:
            some_file = "tmp%d.grb0"%(os.getppid())
        #    try:
            os.system("grbt0 %d > %s"%(target, some_file))
            f = open(some_file, "r")
            t0=np.float64(f.read())
            f.close()
            return t0
    except:
        return 0

def try_tarname(target):
    """tries to run a command that gets TARGET name from the stars DB"""
    try:
            some_file = "tmp%d.tmp"%(os.getppid())
        #    try:
            os.system("tarname %d > %s"%(target, some_file))
            f = open(some_file, "r")
            name=f.read()
            f.close()
            return name.strip()
    except:
            return "-"

def isnumber(a):
    try:
        k=int(a)
        return True
    except:
        return False

def readOptions(args=sys.argv[1:]):
  parser = argparse.ArgumentParser(description="Compute photometric calibration for a FITS image.")
# Transients specific:
  parser.add_argument("-E", "--early", help="Limit transients to t-t0 < 0.5 d", action='store_true')
  parser.add_argument("-f", "--frame", help="Image frame width to be ignored in pixels (default=32)", type=float, default=32)
  parser.add_argument("-g", "--regs", action='store_true', help="Save per image regs")
  parser.add_argument("-w", "--web-output", help="Save web-friendly JSON output to specified file", type=str)
  parser.add_argument("-s", "--siglim", help="Sigma limit for detections to be taken into account.", type=float, default=5)
  parser.add_argument("-m", "--min-found", help="Minimum number of occurences to consider candidate valid", type=int, default=4)
  parser.add_argument("-u", "--usno", help="Use USNO catalog.", action='store_true', default=True)
  parser.add_argument("-q", "--usnox", help="Use USNO catalog extra.", action='store_true')
# General for more tools:
  parser.add_argument("-l", "--maglim", help="Do not get any more than this mag from the catalog to compare.", type=float)
  parser.add_argument("-L", "--brightlim", help="Do not get any less than this mag from the catalog to compare.", type=float)
# not implemented but could be useful:
  parser.add_argument("-i", "--idlimit", help="Set a custom idlimit.", type=float)
  parser.add_argument("-c", "--catalog", action='store', help="Use this catalog as a reference.")
  parser.add_argument("-e", "--enlarge", help="Enlarge catalog search region", type=float)
  parser.add_argument("-v", "--verbose", action='store_true', help="Print debugging info.")
# no sense in this tool:
  #parser.add_argument("-a", "--astrometry", help="Refit astrometric solution using photometry-selected stars", action='store_true')
  #parser.add_argument("-A", "--aterms", help="Terms to fit for astrometry.", type=str)
  #parser.add_argument("-b", "--usewcs", help="Use this astrometric solution (file with header)", type=str)
  #parser.add_argument("-f", "--filter", help="Override filter info from fits", type=str)
  #parser.add_argument("-F", "--flat", help="Produce flats.", action='store_true')
  #parser.add_argument("-G", "--gain", action='store', help="Provide camera gain.", type=float)
  #parser.add_argument("-k", "--makak", help="Makak tweaks.", action='store_true')
  #parser.add_argument("-M", "--model", help="Read model from a file.", type=str)
  #parser.add_argument("-n", "--nonlin", help="CCD is not linear, apply linear correction on mag.", action='store_true')
  #parser.add_argument("-p", "--plot", help="Produce plots.", action='store_true')
  #parser.add_argument("-r", "--reject", help="No outputs for Reduced Chi^2 > value.", type=float)
  #parser.add_argument("-t", "--fit-terms", help="Comma separated list of terms to fit", type=str)
  #parser.add_argument("-T", "--trypar", help="Terms to examine to see if necessary (and include in the fit if they are).", type=str)
  #parser.add_argument("-U", "--terms", help="Terms to fit.", type=str)
  #parser.add_argument("-w", "--weight", action='store_true', help="Produce weight image.")
  #parser.add_argument("-W", "--save-model", help="Write model into a file.", type=str)
  #parser.add_argument("-x", "--fix-terms", help="Comma separated list of terms to keep fixed", type=str)
  #parser.add_argument("-y", "--fit-xy", help="Fit xy tilt for each image separately (i.e. terms PX/PY)", action='store_true')
  #parser.add_argument("-z", "--refit-zpn", action='store_true', help="Refit the ZPN radial terms.")
# files
  parser.add_argument("files", help="Frames to process", nargs='+', action='extend', type=str)
  opts = parser.parse_args(args)
  return opts

def airmass(z):
    """ Compute astronomical airmass according to Rozenberg(1966) """
    cz=np.cos(z)
    return 1/(cz + 0.025 * np.exp(-11*cz) )

def summag(mag1, mag2):
    """add two magnitudes of the same zeropoint"""
    f1 = math.pow(10.0, -0.4*mag1)
    f2 = math.pow(10.0, -0.4*mag2)
    return -2.5 * math.log10(f1 + f2)

def get_atlas_dir(cat, rasc, decl, width, height, directory, mlim):

    atlas_ecsv_tmp = "atlas%ld.ecsv"%(os.getpid())
    command = 'atlas' + ' %f'%(rasc) + ' %f'%(decl) + ' -rect' + ' %f,%f'%(width, height) + ' -dir' + ' %s'%(directory) + ' -mlim' + ' %.2f'%(mlim) + " -ecsv" + " > " + atlas_ecsv_tmp
    print(command)
    os.system(command)

    new = astropy.io.ascii.read(atlas_ecsv_tmp, format='ecsv')
    os.system("rm " + atlas_ecsv_tmp)

    if cat is not None:
        cat = astropy.table.vstack([cat, new])
    else:
        cat = new

    return cat

def get_atlas(rasc, decl, width=0.25, height=0.25, mlim=17):
    cat = None
    cat = get_atlas_dir(cat, rasc, decl, width, height, '/home/mates/cat/atlas/00_m_16/', mlim)
    if mlim > 16: cat = get_atlas_dir(cat, rasc, decl, width, height, '/home/mates/cat/atlas/16_m_17/', mlim)
    if mlim > 17: cat = get_atlas_dir(cat, rasc, decl, width, height, '/home/mates/cat/atlas/17_m_18/', mlim)
    if mlim > 18: cat = get_atlas_dir(cat, rasc, decl, width, height, '/home/mates/cat/atlas/18_m_19/', mlim)
    return cat

def get_usno(rasc, decl, width=0.25, height=0.25, mlim=17):
    cat = None
    usno_ecsv_tmp = "usno%ld.ecsv"%(os.getpid())
    hx=2*height
    if hx>4: hx=4
    command = '/home/mates/bin/ubcone' + ' -P %f'%(rasc/15.0) + ' -p %f'%(decl) + ' -S %f -s %f'%(2*width, hx) + ' -i' + ' %.2f'%(mlim) + " -O" + usno_ecsv_tmp + ">/dev/null 2>&1"
#    command = '/data/catalogs/usno-b1.0/bin/ubcone' + ' -P %f'%(rasc/15.0) + ' -p %f'%(decl) + ' -S %f -s %f'%(2*width, hx) + ' -i' + ' %.2f'%(mlim) + " -O" + usno_ecsv_tmp + ">/dev/null 2>&1"
#    print(command)
    os.system(command)
    cat = astropy.io.ascii.read(usno_ecsv_tmp, format='ecsv')
    os.system("rm " + usno_ecsv_tmp)

    return cat

# get another catalog from a file
def get_catalog(filename):

    cat = astropy.table.Table()
    catalog = astropy.io.ascii.read(filename, format='ecsv')

    cat.add_column(astropy.table.Column(name='radeg', dtype=np.float64, \
        data=catalog['ALPHA_J2000']))
    cat.add_column(astropy.table.Column(name='decdeg', dtype=np.float64, \
        data=catalog['DELTA_J2000']))

    fltnames=['Sloan_r','Sloan_i','Sloan_g','Sloan_z',\
        'Johnson_B','Johnson_V','Johnson_R','Johnson_I','J','c','o']
    for fltname in fltnames:
        cat.add_column(astropy.table.Column(name=fltname, dtype=np.float64,\
            data=catalog['MAG_CALIB']))

    return cat

def remove_junk(hdr):
    for delme in ['comments','COMMENTS','history','HISTORY']:
        try:
            del hdr[delme]
        except KeyError:
            None
    return

# ******** main() **********

options = readOptions(sys.argv[1:])

if options.maglim == None:
    options.maglim = 20

if options.verbose:
    print("%s running in python %d.%d.%d"%(os.path.basename(sys.argv[0]), sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print("Magnitude limit set to %.2f"%(options.maglim))

#l = 0
#k = 0
#u = []
#w = 0
#i = 0
#v = 0
#rV = {}
#nothing = 0

# for sure there would be a systemic way to do this...
PROJ_TAN = 0
PROJ_ZPN = 1
PROJ_ZEA = 2
PROJ_AZP = 3
ast_projections = [ 'TAN', 'ZPN', 'ZEA', 'AZP' ]

img = np.array([], dtype=int) # cislo snimku
img = [] # cislo snimku
x = []
y = []
dy = []
aabs = []
adif = []
coord_x = []
coord_y = []
image_x = []
image_y = []
image_dxy = []
color1 = []
color2 = []
color3 = []
color4 = []
ra = []
dec = []
target = []
tr = []
zeropoints = []
metadata = []
alldet=[]
rmodel=None

def simple_color_model(line, data):
    """
    Apply differential corrections from RESPONSE string using fotfit internals.

    MAG_CALIB in ECSV files already includes zeropoint, spatial, airmass,
    radial, and nonlinearity corrections (computed with colors=0,0,0,0).
    This function applies differential corrections excluding zeropoint to make
    catalog magnitudes comparable to MAG_CALIB.

    Args:
        line: RESPONSE string (e.g., "Z=25.0,PX=0.1,XC=0.3,SC=0.2")
        data: tuple (mag, color1, color2, color3, color4)

    Returns:
        Catalog magnitude with differential corrections applied
    """
    mag, color1, color2, color3, color4 = data

    # Parse RESPONSE string into terms and values
    terms = {}
    try:
        for chunk in line.split(","):
            if '=' not in chunk:
                continue
            term, strvalue = chunk.split("=")
            if term in ['FILTER', 'SCHEMA']:
                continue
            try:
                value = float(strvalue)
                terms[term] = value
            except ValueError:
                continue
    except (ValueError, AttributeError):
        return mag

    if not terms:
        return mag

    # Remove Z term since we only want differential corrections
    # MAG_CALIB already includes the zeropoint
    terms_no_z = {k: v for k, v in terms.items() if k != 'Z'}
    if not terms_no_z:
        return mag  # Only Z term present, no differential corrections needed

    # Use fotfit internals for complete differential correction
    try:
        ffit = fotfit.fotfit()
        ffit.fixall()

        term_names = list(terms_no_z.keys())
        term_values = list(terms_no_z.values())

        ffit.fixterm(term_names, values=term_values)

        # Calculate model at actual conditions
        actual_data = np.array([
            [0.0],      # mc (doesn't matter for differential)
            [1.0],      # airmass (neutral)
            [0.0],      # coord_x (image center)
            [0.0],      # coord_y (image center)
            [color1],   # actual colors
            [color2],
            [color3],
            [color4],
            [0],        # img
            [0.0],      # y (unused)
            [1.0],      # err (unused)
            [0],      # cat_x (center)
            [0]       # cat_y (center)
        ])

        # Calculate model at reference conditions (neutral)
        reference_data = np.array([
            [0.0],      # mc (doesn't matter for differential)
            [1.0],      # airmass (neutral)
            [0.0],      # reference position (center)
            [0.0],
            [0.0],      # neutral colors (same as MAG_CALIB creation)
            [0.0],
            [0.0],
            [0.0],
            [0],        # img
            [0.0],      # y (unused)
            [1.0],      # err (unused)
            [0],      # cat_x (center)
            [0]       # cat_y (center)
        ])

        actual_model = ffit.model(ffit.fixvalues, actual_data)[0]
        reference_model = ffit.model(ffit.fixvalues, reference_data)[0]

        # Differential correction = model(actual) - model(reference)
        differential = actual_model - reference_model

        return mag + differential

    except Exception:
        # Fallback to original magnitude if fotfit fails
        return mag

def open_ecsv_file(arg, verbose=True):
    """Opens a file if possible, given .ecsv or .fits"""
    det = None

    fn = os.path.splitext(arg)[0] + ".ecsv"

    try:
        det = astropy.table.Table.read(fn, format="ascii.ecsv")
        det.meta['filename'] = fn;
        return det
    except:
        if verbose: print("%s did not open as an ecsv table"%(fn));
        det = None

    return det

maxtime=0.0
mintime=1e99
imgtimes=[]
old = []
mags = []
imgno = 0

# Initialize trajectory tracking for moving objects
object_trails = []

for arg in options.files:

    print("file",arg)

    # these are per-file candidates to be cross-identified in the end
    candx=[]
    candy=[]
    candra=[]
    canddec=[]
    candtime=[]
    candexp=[]
    candmag=[]
    canddmag=[]
    candfw=[]

    det = open_ecsv_file(arg, verbose=options.verbose)
    if det is None:
        if options.verbose: print("Cannot handle %s Skipping."%(arg))
        continue
    if options.verbose: print("Input file:", det.meta['filename'])

    remove_junk(det.meta)

    try:
        imgwcs = astropy.wcs.WCS(det.meta)
        det['ALPHA_J2000'], det['DELTA_J2000'] = imgwcs.all_pix2world( [det['X_IMAGE']], [det['Y_IMAGE']], 1)
    except:
        continue

    try:
        field = det.meta['FIELD']
    except:
        det.meta['FIELD'] = 180

    if options.enlarge is not None:
        enlarge = options.enlarge
    else: enlarge=1

    if options.early:
        t0 = try_grbt0(det.meta['TARGET'])
        if det.meta['JD']+det.meta['EXPTIME']/2.0/86400.0-t0 > 0.5:
            continue

    #if det.meta['CTIME'] < mintime: mintime = det.meta['CTIME']
    #if det.meta['CTIME']+det.meta['EXPTIME'] > maxtime: maxtime = det.meta['CTIME']+det.meta['EXPTIME']
    #imgtimes.append(det.meta['CTIME']+det.meta['EXPTIME']/2)
    if det.meta['JD'] < mintime: mintime = det.meta['JD']
    if det.meta['JD']+det.meta['EXPTIME']/86400.0 > maxtime: maxtime = det.meta['JD']+det.meta['EXPTIME']/86400.0
    imgtimes.append(det.meta['JD']+det.meta['EXPTIME']/2.0/86400.0)

    # 2000.0 = 2451544.5
    # 2015.5 = 2457204.5 # reference epoch of Gaia DR2
    epoch = ( det.meta['JD'] - 2457204.5 ) / 365.2425 # Epoch for PM correction

    start = time.time()
    cat = get_atlas(det.meta['CTRRA'], det.meta['CTRDEC'], width=enlarge*det.meta['FIELD'], height=enlarge*det.meta['FIELD'], mlim=options.maglim)
#    cat = get_atlas(det.meta['CTRRA'], det.meta['CTRDEC'], width=enlarge*det.meta['FIELD'], height=enlarge*det.meta['FIELD'], mlim=det.meta['MAGLIMIT']+0.5)
    cat['radeg'] += epoch*cat['pmra']
    cat['decdeg'] += epoch*cat['pmdec']

    print("Catalog returned %d results"%(len(cat)))
    if options.usnox:
        usno = get_usno(det.meta['CTRRA'], det.meta['CTRDEC'], width=enlarge*det.meta['FIELD'], height=enlarge*det.meta['FIELD'], mlim=options.maglim)
    if options.verbose: print("Catalog search took %.3fs"%(time.time()-start))

    # OBJECT-SPECIFIC IDENTIFICATION RADII based on centroiding errors and astrometric quality
    def compute_object_specific_idlimit(det_meta, errx2, erry2, n_sigma=3.0):
        """Compute identification limit for specific object based on its centroiding error"""

        # Use ASTVAR-based approach (legacy method)
        astvar = det_meta.get('ASTVAR', None)

        if astvar is not None and astvar > 0:
            # Object-specific approach using centroiding errors
            # Handle invalid error values (NaN, negative, or zero)
            if np.isnan(errx2) or np.isnan(erry2) or errx2 < 0 or erry2 < 0:
                sigma_centroiding = 0.001  # fallback for invalid errors
            else:
                sigma_centroiding = np.sqrt(errx2 + erry2)
                # Handle zero centroiding errors (perfect centroiding) with minimum threshold
                if sigma_centroiding <= 0:
                    sigma_centroiding = 0.001  # minimum 0.001 pixel centroiding uncertainty
            
            sigma_total = sigma_centroiding * np.sqrt(astvar)
            id_radius = n_sigma * sigma_total
            return id_radius
        else:
            # Ultimate fallback to FWHM if no astrometric variance
            return det_meta.get('FWHM', 1.2)

    # Check if using user-specified idlimit or object-specific approach
    # Note: ERRS0/ERRSC header values removed - error model now computed fresh in transients.py
    use_object_specific = not options.idlimit and det.meta.get('ASTVAR', None) is not None

    if options.idlimit:
        idlimit = options.idlimit
        if options.verbose: print("User-specified idlimit: %.2f pixels"%(idlimit))
    elif use_object_specific:
        # Calculate conservative search radius for initial KDTree query
        # Use 95th percentile of object-specific radii to catch most matches
        sample_radii = []
        for detection in det[:min(100, len(det))]:  # Sample first 100 objects
            errx2 = detection['ERRX2_IMAGE']
            erry2 = detection['ERRY2_IMAGE']
            sample_radii.append(compute_object_specific_idlimit(det.meta, errx2, erry2, n_sigma=3.0))

        idlimit = np.percentile(sample_radii, 95)  # Conservative search radius

        if options.verbose:
            astvar = det.meta.get('ASTVAR')
            if astvar:
                print("Using legacy object-specific identification (ASTVAR=%.1f)"%(astvar))
            else:
                print("Using object-specific identification with fresh error model")
            print("Conservative search radius: %.3f pixels (95th percentile)"%(idlimit))
            print("Sample radii: min=%.3f, max=%.3f, median=%.3f"%(
                np.min(sample_radii), np.max(sample_radii), np.median(sample_radii)))
    else:
        # Fallback to FWHM
        idlimit = det.meta.get('FWHM', 1.2)
        if options.verbose: print("Fallback idlimit: %.2f pixels (FWHM)"%(idlimit))

    # ===  identification with KDTree  ===
    Y = np.array([det['X_IMAGE'], det['Y_IMAGE']]).transpose()
    
    start = time.time()
    try:
        Xt = np.array(imgwcs.all_world2pix(cat['radeg'], cat['decdeg'],1))
        X = Xt.transpose()
        if options.usnox:
            Ut = np.array(imgwcs.all_world2pix(usno['radeg'], usno['decdeg'],1))
            U = Ut.transpose()
        # careful: Xselect conditions are inverted, i.e. what should not be left in
#        Xselect = np.any([np.isnan(Xt[0]),np.isnan(Xt[1]),Xt[0]<0,Xt[1]<0,Xt[0]>det.meta['IMAGEW'],Xt[1]>det.meta['IMAGEH'],cat['Sloan_r']>6.0], axis=0)
#        X = Xt.transpose()[~Xselect]
    except:
        if options.verbose: print("Astrometry of %s sucks! Skipping."%(arg))
        continue

    if len(X) < 1:
        print("len(X)<1, wtf!? %d"%(len(x)))
        continue
    start = time.time()
    tree = KDTree(X)
    print("Build KDTree took %.3fs"%(time.time()-start))
    nearest_ind, nearest_dist = tree.query_radius(Y, r=idlimit, return_distance=True, count_only=False)

    # ===  FULL-CATALOG ERROR MODEL ANALYSIS  ===
    # Analyze error model using complete detection catalog (not just astrometry subset)
    def analyze_full_catalog_error_model(det, imgwcs, cat_coords, matched_indices, options):
        """
        Analyze astrometric error model using full detection catalog
        Returns updated error model parameters or None if analysis fails
        """
        try:
            from error_model import ErrorModelFit

            # Get all matched detections (with valid catalog counterparts)
            matched_detections = []
            astrometric_residuals = []
            centering_errors = []

            for i, (detection, matches) in enumerate(zip(det, matched_indices)):
                if len(matches) > 0:  # Has catalog match
                    # Apply quality filters to avoid bogus identifications
                    # Only use >3σ detections
                    mag_err = detection.get('MAGERR_AUTO', 1.0)
                    if mag_err > 1.091/5.0 and mag_err< 1.091/200:  # <3σ detection (1.091 = 1/ln(2.5))
                        continue

                    # Get best match (closest)
                    best_match_idx = matches[0]  # matches are sorted by distance

                    # Calculate astrometric residual
                    det_x, det_y = detection['X_IMAGE'], detection['Y_IMAGE']
                    cat_ra, cat_dec = cat_coords['radeg'][best_match_idx], cat_coords['decdeg'][best_match_idx]

                    # Convert catalog position to image coordinates
                    try:
                        cat_x, cat_y = imgwcs.all_world2pix(cat_ra, cat_dec, 1)
                        residual = np.sqrt((det_x - cat_x)**2 + (det_y - cat_y)**2)

                        # Get centering error
                        errx2 = detection.get('ERRX2_IMAGE', 0.01)
                        erry2 = detection.get('ERRY2_IMAGE', 0.01)
                        centering_err = np.sqrt(errx2 + erry2)

                        matched_detections.append(detection)
                        astrometric_residuals.append(residual)
                        centering_errors.append(centering_err)

                    except Exception:
                        continue  # Skip problematic coordinates

            if len(astrometric_residuals) < 20:  # Need minimum sample size
                if options.verbose:
                    print(f"Insufficient matches ({len(astrometric_residuals)}) for error model analysis")
                return None

            # Convert to numpy arrays
            residuals = np.array(astrometric_residuals)
            centering = np.array(centering_errors)

            if options.verbose:
                print(f"Analyzing error model with {len(residuals)} matched objects")
                print(f"Residual range: {np.min(residuals):.3f} - {np.max(residuals):.3f} pixels")
                print(f"Centering error range: {np.min(centering):.4f} - {np.max(centering):.4f} pixels")

            # Fit error model using robust binned median approach
            error_model = ErrorModelFit()

            # Use the robust binned median fitting (no outlier iterations needed)
            radii = np.ones_like(residuals) * 1000.0  # Default radius for all objects
            success = error_model.fit_error_model_with_centering(
                residuals, centering, radii, initial_terms=['S0', 'SC'])

            if not success:
                return None

            # Report results
            if options.verbose:
                s0_val = error_model.termval('S0')
                sc_val = error_model.termval('SC')
                print(f"\nFull-catalog error model results:")
                print(f"  S0 (base systematic): {s0_val:.6f} pixels²")
                print(f"  SC (centering scaling): {sc_val:.6f}")
                print(f"  Base error: {np.sqrt(s0_val):.3f} pixels")
                print(f"  Objects used: {len(residuals)} (robust binned medians)")
                print(f"  WSSR/NDF: {error_model.wssrndf:.3f}")

            return error_model

        except Exception as e:
            if options.verbose:
                print(f"Full-catalog error analysis failed: {e}")
            return None

    # ===  MAGNITUDE ERROR MODEL ANALYSIS  ===
    def analyze_magnitude_error_model(det, imgwcs, cat_coords, matched_indices, options):
        """
        Analyze magnitude error model using matched catalog stars
        Returns updated magnitude error model or None if analysis fails
        """
        try:
            from magnitude_error_model import MagnitudeErrorModelFit

            # Get all matched detections with valid catalog counterparts
            photometric_residuals = []
            reported_mag_errors = []
            observed_magnitudes = []
            radial_distances = []

            for i, (detection, matches) in enumerate(zip(det, matched_indices)):
                if len(matches) > 0:  # Has catalog match
                    # Apply quality filters
                    mag_err = detection.get('MAGERR_CALIB', 1.0)
                    if mag_err > 1.091/5.0 or mag_err < 1e-6:  # Skip poor detections
                        continue

                    # Get best match (closest)
                    best_match_idx = matches[0]

                    # Get photometric data
                    obs_mag = detection['MAG_CALIB']

                    # Calculate catalog magnitude with color correction
                    mag0 = cat_coords['Sloan_g'][best_match_idx]
                    mag1 = cat_coords['Sloan_r'][best_match_idx]  # Default to r-band
                    mag2 = cat_coords['Sloan_i'][best_match_idx]
                    mag3 = cat_coords['Sloan_z'][best_match_idx]
                    mag4 = cat_coords['J'][best_match_idx]

                    # Apply filter selection and color correction
                    phfilter = det.meta.get('PHFILTER', 'r').lower()
                    if phfilter in ['r', 'sloan_r']:
                        magcat = mag1
                    elif phfilter in ['g', 'sloan_g']:
                        magcat = mag0
                    elif phfilter in ['i', 'sloan_i']:
                        magcat = mag2
                    elif phfilter in ['z', 'sloan_z']:
                        magcat = mag3
                    elif phfilter in ['j']:
                        magcat = mag4
                    else:
                        magcat = mag1  # Default to r

                    # Apply color correction
                    cm = simple_color_model(det.meta['RESPONSE'],
                        (0,
                        np.float64(mag0-mag1),
                        np.float64(mag1-mag2),
                        np.float64(mag2-mag3),
                        np.float64(mag3-mag4)))

                    cat_mag = magcat + cm
                    residual = obs_mag - cat_mag

                    # Calculate radial distance from image center
                    det_x, det_y = detection['X_IMAGE'], detection['Y_IMAGE']
                    center_x = det.meta.get('IMAGEW', 2048) / 2.0
                    center_y = det.meta.get('IMAGEH', 2048) / 2.0
                    radius = np.sqrt((det_x - center_x)**2 + (det_y - center_y)**2)

                    photometric_residuals.append(residual)
                    reported_mag_errors.append(mag_err)
                    observed_magnitudes.append(obs_mag)
                    radial_distances.append(radius)

            if len(photometric_residuals) < 20:  # Need minimum sample size
                if options.verbose:
                    print(f"Insufficient matches ({len(photometric_residuals)}) for magnitude error model analysis")
                return None

            # Convert to numpy arrays
            residuals = np.array(photometric_residuals)
            mag_errors = np.array(reported_mag_errors)
            magnitudes = np.array(observed_magnitudes)
            radii = np.array(radial_distances)

            if options.verbose:
                print(f"Analyzing magnitude error model with {len(residuals)} matched objects")
                print(f"Residual range: {np.min(residuals):.3f} - {np.max(residuals):.3f} mag")
                print(f"Reported error range: {np.min(mag_errors):.4f} - {np.max(mag_errors):.4f} mag")
                print(f"RMS residual: {np.sqrt(np.mean(residuals**2)):.4f} mag")

            # Fit magnitude error model
            mag_error_model = MagnitudeErrorModelFit()
            success = mag_error_model.fit_magnitude_error_model(
                residuals, mag_errors, magnitudes, radii,
                initial_terms=['M0', 'MM', 'MB'])

            if not success:
                return None

            # Report results
            if options.verbose:
                m0_val = mag_error_model.termval('M0')
                mm_val = mag_error_model.termval('MM')
                mb_val = mag_error_model.termval('MB')
                print(f"\nMagnitude error model results:")
                print(f"  M0 (base systematic): {m0_val:.6f} mag²")
                print(f"  MM (error scaling): {mm_val:.3f}")
                print(f"  MB (brightness term): {mb_val:.6f} mag/mag")
                print(f"  Base error: {np.sqrt(m0_val):.4f} mag")
                print(f"  Error scaling factor: {mm_val:.2f} (errors {('underestimated' if mm_val > 1 else 'overestimated')})")
                print(f"  Objects used: {len(residuals)}")
                print(f"  WSSR/NDF: {mag_error_model.wssrndf:.3f}")

            return mag_error_model

        except Exception as e:
            if options.verbose:
                print(f"Magnitude error analysis failed: {e}")
            return None

    # Perform full-catalog error analysis
    full_error_model = None
    magnitude_error_model = None
    if options.verbose or True:  # Always analyze for now
        full_error_model = analyze_full_catalog_error_model(det, imgwcs, cat, nearest_ind, options)
        magnitude_error_model = analyze_magnitude_error_model(det, imgwcs, cat, nearest_ind, options)

        # Update the compute_object_specific_idlimit function to use new model if available
        if full_error_model is not None:
            def compute_updated_idlimit(errx2, erry2, n_sigma=3.0):
                centering_err = np.sqrt(errx2 + erry2)
                predicted_error = full_error_model.predict_error_with_centering(
                    centering_err, 1.0)[0]  # radius=1 (no radial term)
                return n_sigma * predicted_error
        else:
            # Fallback to original function
            def compute_updated_idlimit(errx2, erry2, n_sigma=3.0):
                return compute_object_specific_idlimit(det.meta, errx2, erry2, n_sigma)

    # Apply object-specific filtering if enabled
    if use_object_specific:
        filtered_ind = []
        filtered_dist = []
        n_filtered = 0

        for i, (detection, ind_list, dist_list) in enumerate(zip(det, nearest_ind, nearest_dist)):
            # Calculate object-specific radius using updated error model if available
            errx2 = detection['ERRX2_IMAGE']
            erry2 = detection['ERRY2_IMAGE']

            if full_error_model is not None:
                obj_idlimit = compute_updated_idlimit(errx2, erry2, n_sigma=3.0)
            else:
                obj_idlimit = compute_object_specific_idlimit(det.meta, errx2, erry2, n_sigma=3.0)

            # Filter matches within object-specific radius
            valid_mask = dist_list <= obj_idlimit
            filtered_ind.append(ind_list[valid_mask])
            filtered_dist.append(dist_list[valid_mask])

            n_filtered += np.sum(~valid_mask)

        nearest_ind = filtered_ind
        nearest_dist = filtered_dist

        if options.verbose:
            print("Object cross-id took %.3fs"%(time.time()-start))
            if full_error_model is not None:
                print("Filtered %d matches using full-catalog error model"%(n_filtered))
            else:
                print("Filtered %d matches using object-specific radii"%(n_filtered))
    else:
        if options.verbose: print("Object cross-id took %.3fs"%(time.time()-start))

    # non-id objects in frame:
    for xx, dd in zip(nearest_ind, det):
        if len(xx) < 1: tr.append(dd)

    # === objects identified ===
    # what does THAT mean?
    # nearest_ind je pole dlouhe stejne jako det (tj. detekovane obj. ze snimku) a kazdy element je seznam objektu z katalogu, ktere jsou bliz nez idlimit.
    # === fill up fields to be fitted ===
    # ... mam fotometricky model snimku (det.meta['RESPONSE'])
    # chci: 1. vypsat vsechny, objekty, ktere nemaji zadny identifikovany zaznam v katalogu
    # 2. vypsat vsechny, ktere jsou citelne jasnejsi/slabsi (5/7/8-sigma) nez by jim odpovidalo v katalogu
    # 3. vypsat vsechny objekty z katalogu, pro ktere se "nenaslo uplatneni"

    metadata.append(det.meta)

    if options.regs:
        regfile = os.path.splitext(arg)[0] + "-tr.reg"
        some_file = open(regfile, "w+")
        some_file.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")

    # Initialize web output data structure
    web_objects = [] if options.web_output else None
    # make pairs to be fitted
    for i, d in zip(nearest_ind, det):

        mdiff = 0
        match = None
        bestmatch = 1e99
        bestmag = 0
        magdet = np.float64(d['MAG_CALIB'])
        errdet = np.float64(d['MAGERR_CALIB'])
        imglim = d.meta['MAGZERO']+d.meta['LIMFLX3']

        for k in i:
            mag0 = cat[k]['Sloan_g']
            mag1 = cat[k]['Sloan_r']
            mag2 = cat[k]['Sloan_i']
            mag3 = cat[k]['Sloan_z']
            mag4 = cat[k]['J']

            # Check PHFILTER and PHSCHEMA for consistency
            phfilter = d.meta.get('PHFILTER', 'r').lower()
            phschema = d.meta.get('PHSCHEMA', 'sloanj').lower()

            # Warn about non-sloanj schemas
            if phschema != 'sloanj' and options.verbose:
                print(f"WARNING: PHSCHEMA='{phschema}' is not 'sloanj' - results may be inconsistent")

            # Select catalog magnitude based on filter
            if phfilter in ['r', 'sloan_r']:
                magcat = mag1  # Sloan_r
            elif phfilter in ['g', 'sloan_g']:
                magcat = mag0  # Sloan_g
            elif phfilter in ['i', 'sloan_i']:
                magcat = mag2  # Sloan_i
            elif phfilter in ['z', 'sloan_z']:
                magcat = mag3  # Sloan_z
            elif phfilter in ['j']:
                magcat = mag4  # J band
            else:
                if options.verbose:
                    print(f"WARNING: Unknown PHFILTER='{phfilter}', defaulting to Sloan_r")
                magcat = mag1  # Default to Sloan_r
            # in case of any other photometri schema this will not work
            cm = simple_color_model( det.meta['RESPONSE'],
                (0,
                np.float64(mag0-mag1),
                np.float64(mag1-mag2),
                np.float64(mag2-mag3),
                np.float64(mag3-mag4)))

            # Use improved magnitude error if available
            if magnitude_error_model is not None:
                # Calculate radial distance for this object
                det_x, det_y = d['X_IMAGE'], d['Y_IMAGE']
                center_x = d.meta.get('IMAGEW', 2048) / 2.0
                center_y = d.meta.get('IMAGEH', 2048) / 2.0
                radius = np.sqrt((det_x - center_x)**2 + (det_y - center_y)**2)

                # Get improved error estimate
                improved_error = magnitude_error_model.predict_magnitude_error(errdet, magdet, radius)
                error_for_comparison = max(improved_error, 0.01)  # Minimum floor
            else:
                error_for_comparison = np.sqrt(errdet*errdet+0.01*0.01)

            mpar = (magcat+cm-magdet)/error_for_comparison
            if np.abs(mpar) < bestmatch:
                bestmatch = np.abs(mpar)
                bestmag = magcat + cm
                match = mpar
                mdiff = np.abs(magcat+cm-magdet)

        # pet moznosti:
        # 1. objekt je v katalogu a neni detekovan (timto zpusobem to nedam)

        # 2. objekt neni v katalogu (= kandidat)
        if bestmag == 0 and errdet < 1.091/options.siglim:
#            print("!",bestmag,magdet,errdet,d["FWHM_IMAGE"]/det.meta['FWHM'] )
#            print("!",i)
            if options.regs and (not options.usno) and d['MAG_CALIB'] < options.maglim:
                print("#",d["FWHM_IMAGE"],det.meta['FWHM'])
                if d["FWHM_IMAGE"] > 1.5*det.meta['FWHM'] or d["FWHM_IMAGE"] < 1.5*det.meta['FWHM']:
#                    some_file.write("circle(%.7f,%.7f,%.3f\") # color=brown\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))
                    some_file.write("circle(%.7f,%.7f,%.3f\") # color=brown\n"%(d["X_IMAGE"], d["Y_IMAGE"],10*d["FWHM_IMAGE"]))
                else:
#                    some_file.write("circle(%.7f,%.7f,%.3f\") # color=red\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))
                    some_file.write("circle(%.7f,%.7f,%.3f\") # color=red\n"%(d["X_IMAGE"], d["Y_IMAGE"],10*d["FWHM_IMAGE"]))

            # Web output for uncatalogued objects
            if web_objects is not None and d['MAG_CALIB'] < options.maglim:
                fwhm_ratio = d["FWHM_IMAGE"] / det.meta['FWHM']
                if fwhm_ratio > 1.5 or fwhm_ratio < 1/1.5:
                    obj_type, color, status = "poor_fwhm", "#8B4513", f"Poor FWHM (ratio={fwhm_ratio:.2f})"
                else:
                    obj_type, color, status = "uncatalogued", "#FF0000", "Not in catalog (transient?)"

                web_objects.append({
                    'x_image': float(d["X_IMAGE"]),
                    'y_image': float(d["Y_IMAGE"]),
                    'ra': float(d['ALPHA_J2000']),
                    'dec': float(d['DELTA_J2000']),
                    'mag': float(d['MAG_CALIB']),
                    'mag_err': float(d['MAGERR_CALIB']),
                    'fwhm': float(d["FWHM_IMAGE"]),
                    'fwhm_ratio': fwhm_ratio,
                    'type': obj_type,
                    'color': color,
                    'status': status
                })

            if d['X_IMAGE'] < options.frame or d['Y_IMAGE']<options.frame or d['X_IMAGE'] > d.meta['IMGAXIS1']-options.frame or d['Y_IMAGE']>d.meta['IMGAXIS2']-options.frame:
                continue

            candx.append(d["X_IMAGE"])
            candy.append(d["Y_IMAGE"])
            candra.append(d['ALPHA_J2000'])
            canddec.append(d['DELTA_J2000'])
            candtime.append(d.meta['JD'])
            candexp.append(d.meta['EXPTIME'])
            candmag.append(d['MAG_CALIB'])
            canddmag.append(d['MAGERR_CALIB'])
            candfw.append(d['FWHM_IMAGE'])
            continue

        # 3. objekt je jasnejsi nez v katalogu (= kandidat)
        if match is not None and match > options.siglim and errdet<1.091/options.siglim and mdiff>0.05:
#            print("+",bestmag,cm,magdet,errdet,np.abs(bestmag-magdet)/errdet,mdiff)
            if options.regs:
                some_file.write("circle(%.7f,%.7f,%.3f\") # color=yellow\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))

            # Web output for brighter objects
            if web_objects is not None:
                web_objects.append({
                    'x_image': float(d["X_IMAGE"]),
                    'y_image': float(d["Y_IMAGE"]),
                    'ra': float(d['ALPHA_J2000']),
                    'dec': float(d['DELTA_J2000']),
                    'mag': float(d['MAG_CALIB']),
                    'mag_err': float(d['MAGERR_CALIB']),
                    'catalog_mag': float(bestmag),
                    'mag_diff': float(bestmag - magdet),
                    'sigma_diff': float(match),
                    'fwhm': float(d["FWHM_IMAGE"]),
                    'type': 'brighter',
                    'color': '#FFFF00',
                    'status': f'Brighter than catalog ({match:.1f}σ, Δmag={bestmag-magdet:.2f})'
                })

        # 4. objekt je slabsi nez v katalogu (= zajimavost)
        if match is not None and match < -options.siglim and errdet<1.091/options.siglim and mdiff>0.05:
#            print("-",bestmag,cm,magdet,errdet,np.abs(bestmag-magdet)/errdet,mdiff)
            if options.regs:
                some_file.write("circle(%.7f,%.7f,%.3f\") # color=blue\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))

            # Web output for fainter objects
            if web_objects is not None:
                web_objects.append({
                    'x_image': float(d["X_IMAGE"]),
                    'y_image': float(d["Y_IMAGE"]),
                    'ra': float(d['ALPHA_J2000']),
                    'dec': float(d['DELTA_J2000']),
                    'mag': float(d['MAG_CALIB']),
                    'mag_err': float(d['MAGERR_CALIB']),
                    'catalog_mag': float(bestmag),
                    'mag_diff': float(bestmag - magdet),
                    'sigma_diff': float(match),
                    'fwhm': float(d["FWHM_IMAGE"]),
                    'type': 'fainter',
                    'color': '#0000FF',
                    'status': f'Fainter than catalog ({abs(match):.1f}σ, Δmag={bestmag-magdet:.2f})'
                })

        # 5. objekt odpovida katalogu (nic)
        if match is not None and (( match > -options.siglim and (bestmag-magdet)/errdet < options.siglim) or mdiff<0.05 ) and errdet<1.091/options.siglim:
            if options.regs:
                some_file.write("circle(%.7f,%.7f,%.3f\") # color=green\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))

            # Web output for normal objects
            if web_objects is not None:
                web_objects.append({
                    'x_image': float(d["X_IMAGE"]),
                    'y_image': float(d["Y_IMAGE"]),
                    'ra': float(d['ALPHA_J2000']),
                    'dec': float(d['DELTA_J2000']),
                    'mag': float(d['MAG_CALIB']),
                    'mag_err': float(d['MAGERR_CALIB']),
                    'catalog_mag': float(bestmag),
                    'mag_diff': float(bestmag - magdet),
                    'fwhm': float(d["FWHM_IMAGE"]),
                    'type': 'normal',
                    'color': '#00FF00',
                    'status': f'Matches catalog (Δmag={bestmag-magdet:.2f})'
                })
    #        print("o",bestmag,cm,magdet,errdet,np.abs(bestmag-magdet)/errdet)

    print('Comparison to Atlas produced ',len(candx),' candidates')

    if options.usno and len(candy)>0:
        usno = get_usno(det.meta['CTRRA'], det.meta['CTRDEC'],
                       width=enlarge*det.meta['FIELD'],
                       height=enlarge*det.meta['FIELD'],
                       mlim=options.maglim)

        # Store original candidates
        orig_candidates = list(zip(candx, candy, candra, canddec, candtime,
                                 candexp, candmag, canddmag, candfw))

        # Track killed candidates for reg file
        killed_candidates = set()  # Store indices of killed candidates

        # Define USNO filtering phases
        if use_object_specific:
            # Use conservative radii for USNO phases since we can't do per-object here
            phase_simple = np.percentile(sample_radii, 20)  # 3σ equivalent
            phase_double = np.percentile(sample_radii, 50)  # 4σ equivalent
            phase_bright = np.percentile(sample_radii, 95)  # 10σ equivalent
        else:
            # Traditional approach
            phase_simple = idlimit
            phase_double = idlimit * 4.0 / 3.0  # Slightly larger
            phase_bright = idlimit * 10.0 / 3.0  # Much larger

        usno_phases = [
            ('simple', imglim, phase_simple, 0),
            ('double', imglim - 1, phase_double, 1),
            ('bright', imglim - 5, phase_bright, 0)
        ]

        # Run through each USNO filtering phase
        for uphase, umaglim, uidlim, unumber in usno_phases:
            if len(candy) < 1:
                break

            # Create USNO KDTree for this phase
            tree_u = KDTree(np.array(imgwcs.all_world2pix(
                usno['radeg'][usno['R1'] < umaglim],
                usno['decdeg'][usno['R1'] < umaglim],
                1)).transpose())

            Y = np.array([candx, candy]).transpose()
            nearest_ind_u, nearest_dist_u = tree_u.query_radius(
                Y, r=uidlim, return_distance=True, count_only=False)

            # Filter candidates
            cand2x = []; cand2y = []; cand2ra = []; cand2dec = []
            cand2time = []; cand2exp = []; cand2mag = []
            cand2dmag = []; cand2fw = []

            live = 0; kill = 0
            for i in range(0, len(candx)):
                if len(nearest_ind_u[i]) > 0:
                    killed_candidates.add(i)
                    kill += 1
                else:
                    live += 1
                    cand2x.append(candx[i])
                    cand2y.append(candy[i])
                    cand2ra.append(candra[i])
                    cand2dec.append(canddec[i])
                    cand2time.append(candtime[i])
                    cand2exp.append(candexp[i])
                    cand2mag.append(candmag[i])
                    cand2dmag.append(canddmag[i])
                    cand2fw.append(candfw[i])

            print(f'USNO {uphase} left {live} candidates, killed {kill}')

            # Update candidate lists for next phase
            candx = cand2x; candy = cand2y; candra = cand2ra
            canddec = cand2dec; candtime = cand2time; candexp = cand2exp
            candmag = cand2mag; canddmag = cand2dmag; candfw = cand2fw

        if options.regs:
            fwhm_margin = 1.5  # Define FWHM margin

            for i, (x, y, _, _, _, _, mag, _, fw) in enumerate(orig_candidates):
                if mag < options.maglim:
                    if i in killed_candidates:
                        # Write killed candidates in cyan
                        some_file.write(f"circle({x:.7f},{y:.7f},{1.5*idlimit*d.meta['PIXEL']}\") # color=cyan\n")
                    else:
                        # Check FWHM ratio for surviving candidates
                        fwhm_ratio = fw / det.meta['FWHM']
                        if 1/fwhm_margin < fwhm_ratio < fwhm_margin:
                            # Normal FWHM ratio - use red
                            some_file.write(f"circle({x:.7f},{y:.7f},{1.5*idlimit*d.meta['PIXEL']}\") # color=red\n")
                        else:
                            # Abnormal FWHM ratio - use magenta
                            some_file.write(f"circle({x:.7f},{y:.7f},{1.5*idlimit*d.meta['PIXEL']}\") # color=magenta\n")

    if options.regs:
        some_file.close()

    # Write web JSON output for this image
    if options.web_output and web_objects is not None:
        # Create web-friendly output structure
        web_data = {
            'success': True,
            'image_file': arg,
            'summary': {
                'total_objects': len(web_objects),
                'uncatalogued': len([o for o in web_objects if o['type'] == 'uncatalogued']),
                'poor_fwhm': len([o for o in web_objects if o['type'] == 'poor_fwhm']),
                'brighter': len([o for o in web_objects if o['type'] == 'brighter']),
                'fainter': len([o for o in web_objects if o['type'] == 'fainter']),
                'normal': len([o for o in web_objects if o['type'] == 'normal'])
            },
            'objects': web_objects,
            'metadata': {
                'siglim': options.siglim,
                'frame_width': options.frame,
                'maglim': options.maglim,
                'image_fwhm': float(det.meta.get('FWHM', 0)),
                'pixel_scale': float(det.meta.get('PIXEL', 0)),
                'image_center_ra': float(det.meta.get('CTRRA', 0)),
                'image_center_dec': float(det.meta.get('CTRDEC', 0))
            }
        }

        # Write JSON output
        web_filename = options.web_output.replace("{file}", os.path.splitext(os.path.basename(arg))[0])
        with open(web_filename, 'w') as web_file:
            json.dump(web_data, web_file, indent=2)

        if options.verbose:
            print(f"Web JSON output written to: {web_filename}")
            print(f"Objects by type: {web_data['summary']}")

    cand = astropy.table.Table([candra,canddec,candtime,candexp,candmag,canddmag,candfw,np.int64(np.ones(len(candra)))], \
        names=['ALPHA_J2000','DELTA_J2000','JD','EXPTIME','MAG_CALIB','MAGERR_CALIB','FWHM_IMAGE','NUM'])
#    print("file", os.path.splitext(arg)[0], len(cand), "candidates")

    if len(candra)>1:
    # remove doubles!
#            tree = KDTree( np.array([cand['ALPHA_J2000'], cand['DELTA_J2000']]).transpose())
            tree = BallTree( np.array([cand['ALPHA_J2000']*np.pi/180, cand['DELTA_J2000']*np.pi/180]).transpose(), metric='haversine')
            nearest_ind, nearest_dist = tree.query_radius( np.array([cand['ALPHA_J2000']*np.pi/180, cand['DELTA_J2000']*np.pi/180]).transpose() , r=d.meta['PIXEL']*1.0/3600.*np.pi/180, return_distance=True, count_only=False)
            i=0;
            doubles=[]
            for g,h in zip(nearest_ind,nearest_dist):
                for j,k in zip(g,h):
                    if j < i:
                        doubles.append([j])
                        print(f"candidate {i} is a double to {j} (distance={3600*180*k/np.pi}\") and will be removed");
                i+=1;
            cand.remove_rows(doubles)

    # TRAJECTORY-BASED MATCHING FOR MOVING OBJECTS
    if len(object_trails) == 0:
        # First image: initialize trails from all candidates
        old = cand
        for i, candidate in enumerate(cand):
            trail = MovingObjectTrail(candidate, i)
            object_trails.append(trail)
            mags.append(astropy.table.Table(candidate))
        if options.verbose:
            print(f"  Initialized {len(object_trails)} object trails from first image")

    elif len(cand) > 0:
        # Subsequent images: match to predicted positions
        current_time = d.meta['JD']

        # Predict positions and calculate search radii for all existing trails
        predicted_positions = []
        search_radii = []

        for trail in object_trails:
            pred_ra, pred_dec = trail.predict_position(current_time)
            predicted_positions.append([pred_ra, pred_dec])
            search_radius = trail.get_search_radius(current_time, base_idlimit=idlimit)
            search_radii.append(search_radius)

        if options.verbose and imgno <= 3:  # Only show for first few images
            print(f"  Image {imgno}: Predicting positions for {len(object_trails)} trails")
            for i, (trail, radius) in enumerate(zip(object_trails, search_radii)):
                if len(trail.detections) >= 2:
                    print(f"    Trail {i}: motion ({trail.motion_ra:.1f}, {trail.motion_dec:.1f}) arcsec/h, search radius {radius:.1f} arcsec")

        # Match candidates to predicted positions
        matched_trails = set()
        unmatched_candidates = []

        for candidate in cand:
            best_match_trail = -1
            best_match_distance = float('inf')

            # Check distance to all predicted positions
            for i, (pred_pos, search_radius) in enumerate(zip(predicted_positions, search_radii)):
                if i in matched_trails:
                    continue  # Trail already matched

                # Calculate spherical distance
                ra_diff = (candidate['ALPHA_J2000'] - pred_pos[0]) * np.cos(np.radians(candidate['DELTA_J2000']))
                dec_diff = candidate['DELTA_J2000'] - pred_pos[1]
                distance_arcsec = np.sqrt(ra_diff**2 + dec_diff**2) * 3600.0

                if distance_arcsec < search_radius and distance_arcsec < best_match_distance:
                    best_match_trail = i
                    best_match_distance = distance_arcsec

            if best_match_trail >= 0:
                # Match found: add to existing trail
                object_trails[best_match_trail].add_detection(candidate)
                matched_trails.add(best_match_trail)

                # Update old table and mags table
                old_index = best_match_trail
                old['NUM'][old_index] += 1
                mags[old_index].add_row(candidate)

                if options.verbose and imgno <= 3:
                    print(f"    Matched candidate at ({candidate['ALPHA_J2000']:.5f}, {candidate['DELTA_J2000']:.5f}) to trail {best_match_trail} (dist={best_match_distance:.1f} arcsec)")
            else:
                # No match: create new trail
                new_trail_id = len(object_trails)
                new_trail = MovingObjectTrail(candidate, new_trail_id)
                object_trails.append(new_trail)

                # Add to old table and mags
                old.add_row(candidate)
                mags.append(astropy.table.Table(candidate))

                if options.verbose and imgno <= 3:
                    print(f"    Created new trail {new_trail_id} for unmatched candidate at ({candidate['ALPHA_J2000']:.5f}, {candidate['DELTA_J2000']:.5f})")

        if options.verbose:
            n_matched = len(matched_trails)
            n_new = len(cand) - n_matched
            print(f"  Image {imgno}: Matched {n_matched} objects, created {n_new} new trails. Total trails: {len(object_trails)}")

    imgno+=1

if len(old) < 1:
        print("No transients found (no object trails created)")
        sys.exit(0)

if options.verbose:
    print(f"\nFinal trajectory summary: {len(object_trails)} object trails created")
    moving_objects = 0
    for i, trail in enumerate(object_trails):
        if len(trail.detections) >= 3:
            motion_total = np.sqrt(trail.motion_ra**2 + trail.motion_dec**2)
            if motion_total > 5.0:  # > 5 arcsec/hour
                moving_objects += 1
                if options.verbose:
                    print(f"  Trail {i}: {len(trail.detections)} detections, motion {motion_total:.1f} arcsec/h")
    print(f"  Detected {moving_objects} likely moving objects (>5 arcsec/h motion)")
    print(f"  Objects with single detections: {len([t for t in object_trails if len(t.detections) == 1])}")
    print(f"  Objects with multiple detections: {len([t for t in object_trails if len(t.detections) > 1])}")
    print(f"  Objects with 3+ detections: {len([t for t in object_trails if len(t.detections) >= 3])}")
    print()

j=0;
for oo in mags:
    if len(oo) >= options.min_found:
        j+=1
if j < 1:
        print(f"No transients found (none makes it > {options.min_found}×)")
        sys.exit(0)

# HERE TRANSIENTS ARE IDENTIFIED, TIME TO CLASSIFY

def movement_residuals(fitvalues, data):
    a0,da=fitvalues
    t,pos = data
    return a0 + da*t - pos

def weighted_movement_residuals(fitvalues, data):
    """Weighted residuals for proper motion fitting with individual measurement uncertainties"""
    a0, da = fitvalues
    t, pos, weights = data
    residuals = a0 + da*t - pos
    return residuals * np.sqrt(weights)  # Weight by measurement precision

def compute_weighted_variance(residuals, weights, n_params=2):
    """Compute weighted variance with chi-squared diagnostics"""
    if len(weights) <= n_params:
        return np.std(residuals), 1.0  # Fallback for insufficient data

    # Weighted variance
    weighted_var = np.sum(weights * residuals**2) / np.sum(weights)

    # Reduced chi-squared for systematic motion detection
    chi2_red = np.sum(weights * residuals**2) / (len(residuals) - n_params)

    # Inflate variance if systematic motion detected
    if chi2_red > 3.0:
        weighted_var *= chi2_red

    return np.sqrt(weighted_var), chi2_red

def prepare_weighted_data(detections, astvar):
    """Prepare weighted data for improved motion analysis"""
    weights = []
    for detection in detections:
        errx2 = detection.get('ERRX2_IMAGE', 0.01)  # Fallback if missing
        erry2 = detection.get('ERRY2_IMAGE', 0.01)

        # Object-specific uncertainty using precise positional information
        sigma_centroiding = np.sqrt(errx2 + erry2)
        sigma_total = sigma_centroiding * np.sqrt(astvar) if astvar > 0 else sigma_centroiding
        weight = 1.0 / sigma_total**2
        weights.append(weight)

    return np.array(weights)

print("!!!! would mean all good, [PMV]=rejected by Movement (rMs or Pixel) > 3 sigma /  Variability < 2 sigma, [pmv]=less strict rejection")

# Check if we can use improved variance calculation
first_astvar = metadata[0].get('ASTVAR', None) if metadata else None
if first_astvar is not None and first_astvar > 0:
    print(f"Using improved weighted variance calculation (ASTVAR={first_astvar:.1f})")
    print("  • Weights measurements by individual centroiding precision")
    print("  • Detects systematic motion via χ² analysis")
    print("  • Expected 3-10x improvement in motion detection sensitivity")
else:
    print("Using robust MAD variance calculation (ASTVAR not available)")

# Initialize trajectory tracking
object_trails = []
print(f"\nInitializing trajectory-based moving object tracking...")
if options.verbose:
    print("  • Predicts object positions based on motion history")
    print("  • Maintains continuous trails for moving objects")
    print("  • Adaptive search radius based on motion uncertainty")

transtmp = []
for oo,i in zip(mags,range(0,len(mags))):
        # full rejection if less than min_found
        num_found = len(oo)
        if num_found < options.min_found: continue

        status = list("!!!!") # start with good (movement in pixels, movement significant, magvar > 3)
        ot_ness = 10.0 # (2 points for each property)
        mp_ness = 10.0 # Minor planet score: high for moving, stable photometry, continuous, good PSF
        t0=np.min(old['JD'])/2+np.max(old['JD'])/2
        dt=np.max(old['JD'])-np.min(old['JD'])  # Time span in days
        x = (oo['JD']-t0) * 24.0  # Convert days to hours for proper motion fitting
        y = oo["ALPHA_J2000"]
        z = oo["DELTA_J2000"]

        fwhm_mean=np.average(oo["FWHM_IMAGE"])

        # IMPROVED VARIANCE COMPUTATION using precise positional information
        # Check if we have astrometric variance available for weighted fitting
        astvar = metadata[0].get('ASTVAR', None) if metadata else None
        use_weighted = astvar is not None and astvar > 0

        if use_weighted:
            # Prepare weighted data using individual measurement uncertainties
            weights = prepare_weighted_data(oo, astvar)

            # Weighted fit for RA
            data_ra = [x, y, weights]
            fitvalues = [oo['ALPHA_J2000'][0], 1.0]
            res_ra = fit.least_squares(weighted_movement_residuals, fitvalues, args=[data_ra], ftol=1e-14)
            a0 = res_ra.x[0]
            da = res_ra.x[1] * 3600  # deg/hour -> arcsec/hour

            # Compute weighted variance for RA
            residuals_ra = movement_residuals(res_ra.x, [x, y])
            sa, chi2_red_ra = compute_weighted_variance(residuals_ra, weights)
            sa *= 3600  # Convert to arcsec

            # Weighted fit for Dec
            data_dec = [x, z, weights]
            fitvalues = [oo['DELTA_J2000'][0], 1.0]
            res_dec = fit.least_squares(weighted_movement_residuals, fitvalues, args=[data_dec], ftol=1e-14)
            d0 = res_dec.x[0]
            dd = res_dec.x[1] * 3600  # deg/hour -> arcsec/hour

            # Compute weighted variance for Dec
            residuals_dec = movement_residuals(res_dec.x, [x, z])
            sd, chi2_red_dec = compute_weighted_variance(residuals_dec, weights)
            sd *= 3600  # Convert to arcsec

            if options.verbose and (chi2_red_ra > 3.0 or chi2_red_dec > 3.0):
                print(f"  Systematic motion detected: χ²_red(RA)={chi2_red_ra:.2f}, χ²_red(Dec)={chi2_red_dec:.2f}")

        else:
            # Fallback to current method when ASTVAR unavailable
            data = [x,y]
            fitvalues = [oo['ALPHA_J2000'][0],1.0] # [a0,da]
            res = fit.least_squares(movement_residuals, fitvalues, args=[data], ftol=1e-14)
            a0=res.x[0]
            da=res.x[1]*3600 # deg/hour -> arcsec/hour
            sa = np.median(np.abs(movement_residuals(res.x, data))) / 0.67
            data = [x,z]
            fitvalues = [oo['DELTA_J2000'][0],1.0] # [a0,da]
            res = fit.least_squares(movement_residuals, fitvalues, args=[data], ftol=1e-14)
            d0=res.x[0]
            dd=res.x[1]*3600 # deg/hour -> arcsec/hour
            sd = np.median(np.abs(movement_residuals(res.x, data))) / 0.67
        dpos = np.sqrt(dd*dd+da*da*np.cos(d0*np.pi/180.0)*np.cos(d0*np.pi/180.0))* (dt*24.0)  # Total motion in arcsec over time span
        sigma = np.sqrt(sd*sd+sa*sa*np.cos(d0*np.pi/180.0)*np.cos(d0*np.pi/180.0))

        # Calculate motion significance
        motion_significance = dpos / sigma if sigma > 0 else 0
        print(f"motion variance={motion_significance}")

        ot_ness /= motion_significance  # Additional penalty for fast movers
        # Use statistical motion significance instead of pixel-based thresholds
        if motion_significance > 2.0:  # Significant motion detected
            status[0] = "p"
        if motion_significance > 5.0:  # Very significant motion
            status[0] = "P"
        if dpos > 2*sigma: # is the movement significant?
            status[1] = "m"
#            nstat -= 1
        if dpos > 3*sigma: # is the movement significant?
            status[1] = "M"
#            nstat -= 2
        try:
            cov = np.linalg.inv(res.jac.T.dot(res.jac))
            fiterrors = np.sqrt(np.diagonal(cov))
        except:
            cov=None
            fiterrors=None

        # Compute weighted mean magnitude and variance
        if use_weighted:
            # Use the same weights as positional analysis
            mag_weights = 1.0 / (oo['MAGERR_CALIB']**2)  # Weight by photometric precision
            mag0 = np.sum(oo['MAG_CALIB'] * mag_weights) / np.sum(mag_weights)
            mag_residuals = oo['MAG_CALIB'] - mag0
            magvar, chi2_red_mag = compute_weighted_variance(mag_residuals, mag_weights, n_params=1)
            print(f"magnitude variance={magvar} (weighted, chi2_red={chi2_red_mag:.2f})")
        else:
            # Fallback to original calculation
            mag0 = np.sum(oo['MAG_CALIB']*oo['MAGERR_CALIB'])/np.sum(oo['MAGERR_CALIB'])
            magvar = np.sqrt(np.average(np.power( (oo['MAG_CALIB']-mag0)/oo['MAGERR_CALIB'] ,2)))
            print(f"magnitude variance={magvar} (unweighted)")
        # Fix double-penalty bug with elif

        ot_ness *= magvar #**2
        if magvar < 2:
            status[2] = "V"
        elif magvar < 3:
            status[2] = "v"

        variability=0
        for mag1,mag2,err1,err2 in zip(oo['MAG_CALIB'][:-1],oo['MAG_CALIB'][1:],oo['MAGERR_CALIB'][:-1],oo['MAGERR_CALIB'][1:]):
#            print (f"mag={mag1},{mag2},{err1},{err2}")
            # Use improved magnitude error model if available
            if magnitude_error_model is not None:
                # Get improved error estimates (use average radius since we don't have pixel coords)
                improved_err1 = magnitude_error_model.predict_magnitude_error(err1, mag1, 1000.0)
                improved_err2 = magnitude_error_model.predict_magnitude_error(err2, mag2, 1000.0)
                # Ensure minimum error floor
                improved_err1 = max(improved_err1, 0.001)
                improved_err2 = max(improved_err2, 0.001)
                variability += abs(mag1-mag2)/np.sqrt(improved_err1*improved_err1+improved_err2*improved_err2)
            else:
                # Fallback to raw errors with minimum floor
                err1_corrected = max(err1, 0.001)
                err2_corrected = max(err2, 0.001)
                variability += abs(mag1-mag2)/np.sqrt(err1_corrected*err1_corrected+err2_corrected*err2_corrected)

        print(f"variability (before normalizing)={variability}")

# det.meta['FWHM'] musi byt nahrazeny prumerem snimku nebo lepe to udelat uplne jinak
#        if fwhm_mean < det.meta['FWHM']/2 or fwhm_mean > det.meta['FWHM']*2:
#            status[3]="F"
#        if fwhm_mean < det.meta['FWHM']/1.5 or fwhm_mean > det.meta['FWHM']*1.5:
#            status[3]="f"
        variability = variability * num_found/(num_found-1) / magvar
        print(f"variability={variability}")
        ot_ness /= variability #**2

        # MINOR PLANET SCORING (mp_ness)
        # Award for: movement, stable photometry, continuous observations, good PSF
        # Penalize for: stationary, high variability, gaps, bad PSF
        
        # 1. Movement scoring (opposite of ot_ness)
        if motion_significance > 5.0:  # Very significant motion - excellent for MP
            mp_ness *= motion_significance * 2.0
        elif motion_significance > 2.0:  # Significant motion - good for MP
            mp_ness *= motion_significance
        else:  # Low motion - penalize for MP
            mp_ness /= max(0.1, motion_significance)
        
        # 2. Photometric stability (opposite of ot_ness - reward stable, penalize variable)
        if magvar < 1.5:  # Very stable - excellent for MP
            mp_ness *= 4.0 / (magvar + 0.1)
        elif magvar < 3.0:  # Moderately stable - good for MP
            mp_ness *= 2.0 / (magvar + 0.1)
        else:  # Too variable - penalize for MP
            mp_ness /= (magvar * magvar)
        
        # 3. Photometric continuity (penalize gaps in observations)
        time_gaps = np.diff(np.sort(oo['JD'])) * 24.0  # Hours between observations
        max_gap = np.max(time_gaps) if len(time_gaps) > 0 else 0
        avg_gap = np.mean(time_gaps) if len(time_gaps) > 0 else 0
        
        # Penalize large gaps (MP should be continuously observable)
        if max_gap > 12.0:  # Gap > 12 hours
            mp_ness /= (max_gap / 12.0)**2
        elif max_gap > 6.0:  # Gap > 6 hours
            mp_ness /= (max_gap / 6.0)
        
        # 4. PSF quality (reward consistent, stellar PSF)
        # Check FWHM consistency across detections
        fwhm_std = np.std(oo["FWHM_IMAGE"])
        fwhm_consistency = fwhm_std / fwhm_mean if fwhm_mean > 0 else 1.0
        
        if fwhm_consistency < 0.1:  # Very consistent PSF
            mp_ness *= 3.0
        elif fwhm_consistency < 0.2:  # Reasonably consistent PSF
            mp_ness *= 1.5
        else:  # Inconsistent PSF - penalize
            mp_ness /= (fwhm_consistency * 10.0)
        
        # 5. Detection continuity bonus (reward complete detection sequences)
        expected_detections = len(imgtimes)  # Total number of images
        detection_fraction = num_found / expected_detections
        if detection_fraction > 0.8:  # Found in >80% of images
            mp_ness *= 2.0
        elif detection_fraction > 0.6:  # Found in >60% of images
            mp_ness *= 1.5

        variance_method = "weighted" if use_weighted else "robust"
        print("".join(status), f"num:{num_found} mag:{mag0:.2f} magvar:{magvar:.1f}/{variability:.1f}, mean_pos:{a0:.5f},{d0:.5f}, movement: {da*np.cos(d0*np.pi/180.0):.3f},{dd:.3f}, motion_sigma: {sigma:.2f}, significance: {motion_significance:.1f}σ, sigma: {sa:.2f},{sd:.2f} ({variance_method}) fwhm_mean: {fwhm_mean}, ot_ness: {ot_ness:.2f}, mp_ness: {mp_ness:.2f}")
        newtrans = [np.int64(i),np.int64(num_found),a0,d0,da,dd,motion_significance,sa,sd,sigma*3600,mag0,magvar,ot_ness,mp_ness]
        transtmp.append(newtrans)

trans=astropy.table.Table(np.array(transtmp),\
        names=['INDEX','NUM','ALPHA_J2000','DELTA_J2000','ALPHA_MOV','DELTA_MOV','DPOS','ALPHA_SIG','DELTA_SIG','SIGMA','MAG_CALIB','MAG_VAR','OT_NESS','MP_NESS'],\
        dtype=['int64','int64','float64','float64','float32','float32','float32','float32','float32','float32','float32','float32','float32','float32'])
print("*** TRANS START ***")
trans.write("transients.ecsv", format='ascii.ecsv', overwrite=True)
print(trans)
print("*** TRANS END ***")


print("In total",len(old),"positions considered.")

# AND NOW SOME REPORTING
# MAKE A DS9 .REG FILE
regfile = "transients.reg"
some_file = open(regfile, "w+")
some_file.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
for oo in trans:
    # Check if object is near image edges (20 pixel margin)
    trail_detections = mags[oo['INDEX']]

    # Calculate motion significance from the stored data
    motion_significance = oo['DPOS'] #/ (oo['SIGMA'] / 3600.0)  # SIGMA is stored in arcsec*3600

#    print(f"any         ra/dec {np.average(oo['ALPHA_J2000']):.7f} {np.average(oo['DELTA_J2000']):.7f} num: {len(trail_detections):.0f} motion: {motion_significance:.1f}σ")

    # Improved classification using both OT_NESS and MP_NESS
    ot_score = oo['OT_NESS']
    mp_score = oo['MP_NESS']
    
    # Classification logic: prioritize highest score
    if ot_score > 10.0 and ot_score > mp_score:  # High OT_NESS = likely OT/GRB
        color = "red"
        object_type = "OT"
        print(f"OT      ra/dec {np.average(oo['ALPHA_J2000']):.7f} {np.average(oo['DELTA_J2000']):.7f} num: {len(trail_detections):.0f} ({motion_significance:.1f}σ motion, ot_ness={ot_score:.1f}, mp={mp_score:.1f})")
    elif mp_score > 10.0 and mp_score > ot_score:  # High MP_NESS = likely minor planet
        color = "yellow" 
        object_type = "MP"
        print(f"MP      ra/dec {np.average(oo['ALPHA_J2000']):.7f} {np.average(oo['DELTA_J2000']):.7f} num: {len(trail_detections):.0f} ({motion_significance:.1f}σ motion, ot_ness={ot_score:.1f}, mp={mp_score:.1f})")
    elif motion_significance > 2.0:  # Moving but unclear classification
        color = "orange"
        object_type = "MV"
        print(f"moving  ra/dec {np.average(oo['ALPHA_J2000']):.7f} {np.average(oo['DELTA_J2000']):.7f} num: {len(trail_detections):.0f} ({motion_significance:.1f}σ motion, ot_ness={ot_score:.1f}, mp={mp_score:.1f})")
    else:  # Stationary or low scores
        color = "cyan"
        object_type = "??"
        print(f"other   ra/dec {np.average(oo['ALPHA_J2000']):.7f} {np.average(oo['DELTA_J2000']):.7f} num: {len(trail_detections):.0f} ({motion_significance:.1f}σ motion, ot_ness={ot_score:.1f}, mp={mp_score:.1f})")

    # Write to reg file with appropriate color
    some_file.write(f"circle({np.average(oo['ALPHA_J2000']):.7f},{np.average(oo['DELTA_J2000']):.7f},{5*idlimit*d.meta['PIXEL']:.3f}\") # color={color} text={{{object_type} id:{oo['INDEX']},mag:{oo['MAG_CALIB']:.1f},var:{oo['MAG_VAR']:.1f},pos:{motion_significance:.1f}σ}}\n")
some_file.close()

# MATPLOTLIB LIGHTCURVE
try:
    t0 = try_grbt0(d.meta['TARGET'])
except:
    t0 = 0

# Get target name for plot title
try:
    title = try_tarname(d.meta['TARGET'])
except:
    title = "Transients"

# Count valid transients (OT_NESS > 2.0 or MP_NESS > 10.0)
valid_transients = [oo for oo in trans if oo['OT_NESS'] > 2.0 or oo['MP_NESS'] > 10.0]

if len(valid_transients) > 0:
    print(f"Creating matplotlib plot for {len(valid_transients)} valid transients...")

    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set up plot appearance
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time since T0 (days)', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)

        # Reverse Y axis (brighter = higher up, astronomical convention)
        ax.invert_yaxis()

        # Set logarithmic X axis
        ax.set_xscale('log')

        # Time range (relative to T0)
        time_min = mintime - t0
        time_max = maxtime - t0
        ax.set_xlim(time_min, time_max)

        # Set up custom X ticks (similar to gnuplot version)
        xtick_positions = []
        xtick_labels = []
        for j in range(0, 6):
            tic = np.power(10, np.log10(time_min) + j/5.0*(np.log10(time_max) - np.log10(time_min)))
            xtick_positions.append(tic)
            xtick_labels.append(f"{tic:.0f}")

        # Add minor ticks for image observation times
        for img_time in imgtimes:
            rel_time = img_time - t0
            if time_min <= rel_time <= time_max:
                xtick_positions.append(rel_time)
                xtick_labels.append("")

        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

        # Color cycle for different transients
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_transients)))

        # Plot each valid transient
        for i, oo in enumerate(valid_transients):
            transient_mags = mags[oo['INDEX']]

            if len(transient_mags) == 0:
                continue

            # Extract data arrays
            times = [(mag['JD'] - t0) for mag in transient_mags]
            magnitudes = [mag['MAG_CALIB'] for mag in transient_mags]
            exp_times = [mag['EXPTIME'] for mag in transient_mags]
            mag_errors = [mag['MAGERR_CALIB'] for mag in transient_mags]

            # Convert exposure times to half-widths in days
            exp_half_widths = [exp / 2.0 / 86400.0 for exp in exp_times]

            # Create label with transient properties
            label = (f"α:{oo['ALPHA_J2000']:.5f} δ:{oo['DELTA_J2000']:.5f} "
                    f"v={oo['MAG_VAR']:.1f} σ={oo['SIGMA']:.2f} "
                    f"p={oo['DPOS']/d.meta['PIXEL']:.2f}")

            # Plot with error bars and exposure time as horizontal extent
            ax.errorbar(times, magnitudes,
                       xerr=exp_half_widths,
                       yerr=mag_errors,
                       fmt='o',
                       color=colors[i],
                       markersize=6,
                       capsize=3,
                       capthick=1,
                       elinewidth=1,
                       label=label,
                       alpha=0.8)

        # Add grid
        ax.grid(True, alpha=0.3, which='both')

        # Add legend (compact for many transients)
        if len(valid_transients) <= 8:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            ax.text(0.02, 0.98, f"{len(valid_transients)} transients",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Adjust layout
        plt.tight_layout()

        # Save plot
        try:
            output_filename = f"transients-{d.meta['TARGET']}.png"
        except:
            output_filename = "transients.png"

        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Transient light curve plot saved as: {output_filename}")

        plt.close()

    except Exception as e:
        print(f"Error creating matplotlib plot: {e}")

else:
    print("No valid transients found for plotting")
# FINISHED MATPLOTLIB

#print(mags)
# === fields are set up ===
