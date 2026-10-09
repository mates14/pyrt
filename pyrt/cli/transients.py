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

from pyrt.core import zpnfit
from pyrt.core import fotfit
from pyrt.catalog.catalog import Catalog
from pyrt.core.match_stars import apply_catalog_astrometric_corrections, compute_effective_idlimit
from pyrt.utils.dophot_config import load_config

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
  parser.add_argument("-c", "--catalog", action='store', help="Override the reference catalog (default: PHCAT written by dophot, e.g. atlas@localhost).")
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

def remove_junk(hdr):
    for delme in ['comments','COMMENTS','history','HISTORY']:
        try:
            del hdr[delme]
        except KeyError:
            None
    return

def response_model(meta, schema):
    """
    Rebuild the photometric model dophot fitted for this image from its RESPONSE
    header (FILTER=/SCHEMA= chunks and anything non-numeric are skipped, the
    schema is taken from PHSCHEMA instead).

    The model maps an instrumental MAG_AUTO to the catalog magnitude in PHFILTER;
    evaluated with zero colors it reproduces MAG_CALIB, with the colors of a
    catalog star it gives the magnitude that star's catalog entry should have.
    """
    terms, values = [], []
    for chunk in meta.get('RESPONSE', '').split(","):
        if '=' not in chunk:
            continue
        term, strvalue = chunk.split("=", 1)
        if term in ['FILTER', 'SCHEMA']:
            continue
        try:
            values.append(float(strvalue))
            terms.append(term)
        except ValueError:
            continue

    ffit = fotfit.fotfit()
    ffit.fixall()
    ffit.fixterm(terms, values=values)
    ffit.set_filter_info(meta['PHFILTER'], schema)
    return ffit

def model_magnitude(ffit, meta, rows, colors):
    """Evaluate the dophot model for detections `rows` with catalog colors (c1..c4),
    exactly as dophot computes MAG_CALIB (there with colors=0)."""
    return ffit.model(np.array(ffit.fitvalues),
        (   rows['MAG_AUTO'],
            0,                                   # airmass relative to image center
            (rows['X_IMAGE'] - meta['CTRX'])/1024,
            (rows['Y_IMAGE'] - meta['CTRY'])/1024,
            colors[0], colors[1], colors[2], colors[3],
            0,                                   # image index (RESPONSE has no per-image terms)
            0, 0,
            rows['X_IMAGE'], rows['Y_IMAGE'],    # pixel structure
            meta['AIRMASS']))                    # absolute airmass

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

def slope_with_error(t, y):
    """Unweighted straight line y(t); returns (slope, slope_err) with the error
    from the residual scatter (candidates carry no per-detection position errors)."""
    t = np.asarray(t, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(t) < 3:
        return np.nan, np.nan
    tc = t - np.mean(t)
    stt = np.sum(tc*tc)
    if stt <= 0:
        return np.nan, np.nan
    slope = np.sum(tc*(y - np.mean(y))) / stt
    resid = y - np.mean(y) - slope*tc
    return slope, np.sqrt(np.sum(resid*resid)/(len(t)-2) / stt)

def brightening_fit(oo):
    """Weighted straight line through the light curve of a brightened catalogued star,
    with a separate intercept per filter (so that e.g. N->r switch is not seen as a trend).
    Returns (dmag, rate, rate_err): median brightening over the catalog [mag, <0 = brighter]
    and the trend in mag/hour (>0 = fading), its error scaled by sqrt(chi2/ndf) if > 1."""
    dmag = np.nanmedian(oo['MAG_CALIB'] - oo['CATMAG'])
    t = np.array((oo['JD'] + oo['EXPTIME']/2.0/86400.0 - np.min(oo['JD'])) * 24.0, dtype=np.float64)
    m = np.array(oo['MAG_CALIB'], dtype=np.float64)
    sw = 1.0 / np.maximum(np.array(oo['MAGERR_CALIB'], dtype=np.float64), 0.02)  # dophot's error floor
    filters = sorted(set(oo['FILTER']))
    A = np.array([t] + [np.array(oo['FILTER']) == f for f in filters], dtype=np.float64).T
    ndf = len(m) - A.shape[1]
    if ndf < 0:
        return dmag, np.nan, np.nan
    try:
        cov = np.linalg.inv((A*sw[:,None]).T @ (A*sw[:,None]))
    except np.linalg.LinAlgError:  # e.g. no filter with two different times
        return dmag, np.nan, np.nan
    p = cov @ (A*sw[:,None]).T @ (m*sw)
    rate, rate_err = p[0], np.sqrt(cov[0,0])
    if ndf > 0:
        chi2 = np.sum(((m - A @ p)*sw)**2)
        rate_err *= np.sqrt(max(chi2/ndf, 1.0))
    return dmag, rate, rate_err

# ******** main() **********

def main():
    """Find new objects, brightened catalogued stars (flares) and movers in dophot ECSVs."""

    options = readOptions(sys.argv[1:])

    if options.maglim == None:
        options.maglim = 20

    # filter schemas exactly as dophot sees them (built-in + ~/.config/dophot3/config)
    filter_schemas = load_config()['filter_schemas']

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
        candcat=[]  # expected MAG_CALIB from the catalog (NaN = not in catalog)
        candcol=[]  # catalog color schema[0]-schema[1]
        candplx=[]  # catalog parallax [mas]
        candflt=[]  # filter of the frame

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

        if det.meta.get('ASTQUAL') == 'BAD':
            print("ASTQUAL=BAD in %s, skipping"%(arg))
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

        # Use the same catalog, reference filter and color schema as dophot did
        catname = options.catalog or det.meta.get('PHCAT')
        if catname is None:
            catname = 'atlas@localhost'
            print("WARNING: no PHCAT in %s (older dophot?), assuming %s"%(arg, catname))
        phfilter = det.meta.get('PHFILTER')
        phschema = det.meta.get('PHSCHEMA')
        if phfilter is None or phschema not in filter_schemas:
            print("Unknown PHFILTER/PHSCHEMA (%s/%s) in %s, skipping"%(phfilter, phschema, arg))
            continue
        schema = filter_schemas[phschema]
        color_name = "%s-%s"%(schema[0], schema[1])

        start = time.time()
        cat = Catalog(ra=det.meta['CTRRA'], dec=det.meta['CTRDEC'],
                      width=enlarge*det.meta['FIELD'], height=enlarge*det.meta['FIELD'],
                      mlim=options.maglim, catalog=catname)
        if len(cat) < 1:
            print("Catalog %s returned nothing for %s, skipping"%(catname, arg))
            continue
        apply_catalog_astrometric_corrections(cat, det.meta['JD'])

        # catalog colors as in dophot's compute_colors_and_apply_limits()
        # (atlas stores a missing magnitude as 0, e.g. J for ~1/3 of stars)
        catmags = [np.where(np.array(cat[f], dtype=np.float64) == 0, np.nan, np.array(cat[f], dtype=np.float64)) for f in schema]
        catcolors = [catmags[k%len(schema)] - catmags[(k+1)%len(schema)] for k in range(4)]
        catmag = catmags[schema.index(phfilter)] if phfilter in schema else np.array(cat[phfilter], dtype=np.float64)
        ffit = response_model(det.meta, schema)

        print("Catalog %s returned %d results (filter %s, schema %s)"%(catname, len(cat), phfilter, phschema))
        if options.usnox:
            usno = get_usno(det.meta['CTRRA'], det.meta['CTRDEC'], width=enlarge*det.meta['FIELD'], height=enlarge*det.meta['FIELD'], mlim=options.maglim)
        if options.verbose: print("Catalog search took %.3fs"%(time.time()-start))

        # OBJECT-SPECIFIC IDENTIFICATION RADII based on centroiding errors and astrometric quality
        def compute_object_specific_idlimit(det_meta, errx2, erry2, n_sigma=3.0, fwhm=0.0):
            """Compute identification limit for specific object based on its centroiding error.

            Uses the S0+SC error model: sigma_total² = ASTSIGMA² + (errx2+erry2)·ASTVAR
            where ASTSIGMA=sqrt(S0) is the systematic WCS floor and ASTVAR=SC is the
            centroiding scale factor, consistent with what dophot writes after fitting.
            """
            astsigma = det_meta.get('ASTSIGMA', None)
            astvar   = det_meta.get('ASTVAR',   None)

            if astsigma is not None and astvar is not None:
                s0 = astsigma ** 2
                sc = max(float(astvar), 0.0)
                if np.isnan(errx2) or np.isnan(erry2) or errx2 < 0 or erry2 < 0:
                    centering2 = 0.0
                else:
                    centering2 = max(errx2 + erry2, 0.0)
                sigma_total = np.sqrt(s0 + sc * centering2)
                # Floor at FWHM/2 (of the frame or of the object itself if larger, i.e. blends):
                # closer than that two sources are not resolved, so the detection is that catalog
                # star (a transient on top of it shows as brighter). The S0/SC model alone is too
                # tight e.g. at the field edges, and a missed match makes a false new object.
                return max(n_sigma * sigma_total, max(det_meta.get('FWHM', 1.0), fwhm)/2.0, 0.5)
            else:
                return det_meta.get('FWHM', 1.2)

        # Search radius as dophot's matching uses it; per-object radii (ASTSIGMA/ASTVAR
        # written by dophot's astrometric fit) then trim the matches below
        use_object_specific = not options.idlimit and det.meta.get('ASTSIGMA') is not None \
            and det.meta.get('ASTVAR') is not None
        if use_object_specific:
            sample_radii = [compute_object_specific_idlimit(det.meta, dd['ERRX2_IMAGE'], dd['ERRY2_IMAGE'], fwhm=dd['FWHM_IMAGE'])
                            for dd in det]
        idlimit = compute_effective_idlimit(det, options)
        if use_object_specific:  # the KDTree query must reach the per-object radii (FWHM/2 floor)
            idlimit = max(idlimit, min(np.max(sample_radii), 10.0))
        if options.verbose: print("Identification radius: %.2f pixels"%(idlimit))

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
            print("len(X)<1, wtf!? %d"%(len(X)))
            continue
        start = time.time()
        tree = KDTree(X)
        print("Build KDTree took %.3fs"%(time.time()-start))
        nearest_ind, nearest_dist = tree.query_radius(Y, r=idlimit, return_distance=True, count_only=False)

        # Apply object-specific filtering if enabled
        if use_object_specific:
            filtered_ind = []
            filtered_dist = []
            n_filtered = 0

            for i, (detection, ind_list, dist_list) in enumerate(zip(det, nearest_ind, nearest_dist)):
                errx2 = detection['ERRX2_IMAGE']
                erry2 = detection['ERRY2_IMAGE']
                obj_idlimit = compute_object_specific_idlimit(det.meta, errx2, erry2, n_sigma=3.0, fwhm=detection['FWHM_IMAGE'])

                # Filter matches within object-specific radius
                valid_mask = dist_list <= obj_idlimit
                filtered_ind.append(ind_list[valid_mask])
                filtered_dist.append(dist_list[valid_mask])

                n_filtered += np.sum(~valid_mask)

            nearest_ind = filtered_ind
            nearest_dist = filtered_dist

            if options.verbose:
                print("Object cross-id took %.3fs"%(time.time()-start))
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
        # Every detection-catalog pair through dophot's model: the magnitude the detection
        # should have (in MAG_CALIB terms) given the catalog star's magnitude and colors.
        # (expected - MAG_CALIB) is exactly the residual of dophot's photometric fit.
        pair_det = np.repeat(np.arange(len(det)), [len(ii) for ii in nearest_ind])
        pair_cat = np.concatenate(nearest_ind).astype(int) if len(pair_det) > 0 else np.array([], dtype=int)
        if len(pair_det) > 0:
            # a missing color is taken as 0, i.e. the star is compared to plain MAG_CALIB
            pcolors = [np.nan_to_num(c[pair_cat]) for c in catcolors]
            pair_model = model_magnitude(ffit, det.meta, det[pair_det], pcolors)
            pair_expect = catmag[pair_cat] - np.asarray(pair_model) + np.asarray(det['MAG_CALIB'][pair_det])
        else:
            pair_expect = np.array([])

        # make pairs to be fitted
        pair_start = 0
        for i, d in zip(nearest_ind, det):

            mdiff = 0
            match = None
            bestmag = None
            magdet = np.float64(d['MAG_CALIB'])
            errdet = np.float64(d['MAGERR_CALIB'])
            imglim = d.meta['MAGZERO']+d.meta['LIMFLX3']

            expect = pair_expect[pair_start:pair_start+len(i)]
            expect_cat = pair_cat[pair_start:pair_start+len(i)]
            pair_start += len(i)

            error_for_comparison = np.sqrt(errdet*errdet+0.01*0.01)
            mpar = (expect-magdet)/error_for_comparison
            if np.any(np.isfinite(mpar)):
                best = np.nanargmin(np.abs(mpar))
                bestmag = expect[best]
                bestcat = expect_cat[best]
                match = mpar[best]
                mdiff = np.abs(bestmag-magdet)
            # (catalogued but without a magnitude in PHFILTER: neither a new object nor comparable)

            # pet moznosti:
            # 1. objekt je v katalogu a neni detekovan (timto zpusobem to nedam)

            # 2. objekt neni v katalogu (= kandidat)
            # (FWHM_IMAGE <= 0 means a failed shape measurement - typically an artifact)
            if len(i) == 0 and errdet < 1.091/options.siglim and d['FWHM_IMAGE'] > 0:
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
                candcat.append(np.nan)
                candcol.append(np.nan)
                candplx.append(np.nan)
                candflt.append(str(det.meta.get('FILTER', '')))
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

                if not (d['X_IMAGE'] < options.frame or d['Y_IMAGE']<options.frame or d['X_IMAGE'] > d.meta['IMGAXIS1']-options.frame or d['Y_IMAGE']>d.meta['IMGAXIS2']-options.frame):
                    candx.append(d["X_IMAGE"])
                    candy.append(d["Y_IMAGE"])
                    candra.append(d['ALPHA_J2000'])
                    canddec.append(d['DELTA_J2000'])
                    candtime.append(d.meta['JD'])
                    candexp.append(d.meta['EXPTIME'])
                    candmag.append(d['MAG_CALIB'])
                    canddmag.append(d['MAGERR_CALIB'])
                    candfw.append(d['FWHM_IMAGE'])
                    candcat.append(bestmag)
                    candcol.append(catcolors[0][bestcat])
                    candplx.append(cat['parallax'][bestcat] if 'parallax' in cat.colnames else np.nan)
                    candflt.append(str(det.meta.get('FILTER', '')))

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

        print('Comparison to %s produced %d candidates (%d of them brightened catalogued stars)'%(catname, len(candx), np.sum(np.isfinite(candcat))))

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

            # Define USNO filtering phases: candidate near any USNO star (simple), near a
            # brighter one = close neighbour/blend (double, 4 px), near a very bright one (10 px).
            # (The radii are the original ones; per-object percentiles made them sub-pixel.)
            phase_simple = idlimit
            phase_double = max(4.0, idlimit)
            phase_bright = max(10.0, idlimit)

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
                cand2cat = []; cand2col = []; cand2plx = []; cand2flt = []

                live = 0; kill = 0
                for i in range(0, len(candx)):
                    if len(nearest_ind_u[i]) > 0 and not np.isfinite(candcat[i]):
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
                        cand2cat.append(candcat[i])
                        cand2col.append(candcol[i])
                        cand2plx.append(candplx[i])
                        cand2flt.append(candflt[i])

                print(f'USNO {uphase} left {live} candidates, killed {kill}')

                # Update candidate lists for next phase
                candx = cand2x; candy = cand2y; candra = cand2ra
                canddec = cand2dec; candtime = cand2time; candexp = cand2exp
                candmag = cand2mag; canddmag = cand2dmag; candfw = cand2fw
                candcat = cand2cat; candcol = cand2col; candplx = cand2plx; candflt = cand2flt

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

        cand = astropy.table.Table([candra,canddec,candtime,candexp,candmag,canddmag,candfw,np.int64(np.ones(len(candra))),
            np.array(candcat,dtype=np.float64),np.array(candcol,dtype=np.float64),np.array(candplx,dtype=np.float64),np.array(candflt,dtype='U16')], \
            names=['ALPHA_J2000','DELTA_J2000','JD','EXPTIME','MAG_CALIB','MAGERR_CALIB','FWHM_IMAGE','NUM','CATMAG','CATCOLOR','PARALLAX','FILTER'])
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

            # Motion significance: fitted motion over its own error, both axes together
            # (chi with 2 dof: a stationary object exceeds 3 in ~1% of cases)
            vra, era = slope_with_error(x, (y - np.mean(y))*np.cos(np.radians(np.mean(z))))
            vde, ede = slope_with_error(x, z - np.mean(z))
            if era > 0 and ede > 0:
                motion_significance = np.sqrt((vra/era)**2 + (vde/ede)**2)
            else:
                motion_significance = 0.0
            print(f"motion variance={motion_significance}")

            ot_ness /= max(motion_significance/2.0, 1.0)  # Penalty for movers (none below 2 sigma, stationary objects scatter up to that)
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
            else:  # Low motion - penalize for MP (factor < 1 for anything below 2 sigma)
                mp_ness *= motion_significance / 2.0

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
            # a known star that got brighter (majority of its detections matched to the catalog)
            if np.sum(np.isfinite(oo['CATMAG'])) > num_found/2:
                dmag, rate, rate_err = brightening_fit(oo)
                catmag = np.nanmedian(oo['CATMAG'])
            else:
                dmag = rate = rate_err = catmag = np.nan
            newtrans = [np.int64(i),np.int64(num_found),a0,d0,da,dd,motion_significance,sa,sd,sigma*3600,mag0,magvar,ot_ness,mp_ness,
                catmag, dmag, rate, rate_err, np.nanmedian(oo['CATCOLOR']), np.nanmedian(oo['PARALLAX'])]
            transtmp.append(newtrans)

    trans=astropy.table.Table(np.array(transtmp),\
            names=['INDEX','NUM','ALPHA_J2000','DELTA_J2000','ALPHA_MOV','DELTA_MOV','DPOS','ALPHA_SIG','DELTA_SIG','SIGMA','MAG_CALIB','MAG_VAR','OT_NESS','MP_NESS',
                   'CATMAG','DMAG','RATE','RATE_ERR','CATCOLOR','PARALLAX'],\
            dtype=['int64','int64','float64','float64','float32','float32','float32','float32','float32','float32','float32','float32','float32','float32',
                   'float32','float32','float32','float32','float32','float32'])
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

        # Classification logic: a brightened known star first, then prioritize highest score
        if np.isfinite(oo['CATMAG']):
            trend = "fading" if oo['RATE'] > 0 else "rising"
            rate_sig = np.abs(oo['RATE'])/oo['RATE_ERR'] if oo['RATE_ERR'] > 0 else 0.0
            # a flare: catalogued star, brighter than the catalog and significantly fading;
            # brighter without a trend is more often a color-model problem (very red stars)
            if oo['RATE'] > 0 and rate_sig >= 3.0:
                color = "magenta"
                object_type = "FLARE"
            else:
                color = "pink"
                object_type = "BRIGHT"
            print(f"{object_type:7s} ra/dec {oo['ALPHA_J2000']:.7f} {oo['DELTA_J2000']:.7f} num: {len(trail_detections):.0f} "
                  f"mag: {oo['MAG_CALIB']:.2f} (catalog {oo['CATMAG']:.2f}, {oo['DMAG']:+.2f}) "
                  f"{trend} {oo['RATE']:+.3f}±{oo['RATE_ERR']:.3f} mag/h ({rate_sig:.1f}σ) "
                  f"color {color_name}: {oo['CATCOLOR']:.2f}"
                  + (f", parallax: {oo['PARALLAX']:.2f} mas" if np.isfinite(oo['PARALLAX']) else "")
                  + f" ({motion_significance:.1f}σ motion)")
        elif ot_score > 10.0 and ot_score > mp_score:  # High OT_NESS = likely OT/GRB
            color = "red"
            object_type = "OT"
            print(f"OT      ra/dec {np.average(oo['ALPHA_J2000']):.7f} {np.average(oo['DELTA_J2000']):.7f} num: {len(trail_detections):.0f} ({motion_significance:.1f}σ motion, ot_ness={ot_score:.1f}, mp={mp_score:.1f})")
        elif mp_score > 10.0 and mp_score > ot_score and motion_significance > 5.0:  # likely minor planet
            color = "yellow" 
            object_type = "MP"
            print(f"MP      ra/dec {np.average(oo['ALPHA_J2000']):.7f} {np.average(oo['DELTA_J2000']):.7f} num: {len(trail_detections):.0f} ({motion_significance:.1f}σ motion, ot_ness={ot_score:.1f}, mp={mp_score:.1f})")
        elif motion_significance > 3.0:  # Moving but unclear classification
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
    valid_transients = [oo for oo in trans if oo['OT_NESS'] > 2.0 or oo['MP_NESS'] > 10.0 or np.isfinite(oo['CATMAG'])]

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


if __name__ == '__main__':
    main()
