#!/usr/bin/env python3

import os
import sys
import time
import argparse
from contextlib import suppress

import numpy as np

import astropy
import astropy.io.fits
import astropy.wcs
import astropy.table
from astropy.coordinates import SkyCoord

# This is to silence a particular annoying warning (MJD not present in a fits file)
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

from sklearn.neighbors import KDTree

import zpnfit
import fotfit

from cat2det import try_img,remove_junk
from refit_astrometry import refit_astrometry

if sys.version_info[0]*1000+sys.version_info[1]<3008:
    print(f"Error: python3.8 or higher is required (this is python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]})")
    sys.exit(-1)

def exportColumnsForDS9(columns, file="ds9.reg", size=10, width=3, color="red"):
    some_file = open(file, "w+")
    some_file.write(f"# Region file format: DS9 version 4.1\nglobal color={color} dashlist=8 3 width={width}"\
        " font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
    for a, d in zip(columns[0], columns[1]):
        some_file.write(f"circle({a:.7f},{d:.7f},{size:.3f}\") # color={color} width={width}\n")

def readOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Compute photometric calibration for a FITS image.")
    parser.add_argument("-a", "--astrometry", help="Refit astrometric solution using photometry-selected stars", action='store_true')
    parser.add_argument("-A", "--aterms", help="Terms to fit for astrometry.", type=str)
    parser.add_argument("--usewcs", help="Use this astrometric solution (file with header)", type=str)
    parser.add_argument("-b", "--basemag", help="ID of the base filter to be used while fitting (def=\"Sloan_r\"/\"Johnson_V\")", type=str)
    parser.add_argument("-c", "--catalog", action='store', help="Use this catalog as a reference.")
    parser.add_argument("-d", "--date", action='store', help="what to put into the third column (char,mid,bjd), default=mid")
    parser.add_argument("-e", "--enlarge", help="Enlarge catalog search region", type=float)
    parser.add_argument("-f", "--filter", help="Override filter info from fits", type=str)
    parser.add_argument("-F", "--flat", help="Produce flats.", action='store_true')
    parser.add_argument("-g", "--guessbase", action='store_true', help="Try and set base filter from fits header (implies -j if Bessel filter is found).")
    parser.add_argument("-X", "--tryflt", action='store_true', help="Try different filters (broken).")
    parser.add_argument("-G", "--gain", action='store', help="Provide camera gain.", type=float)
    parser.add_argument("-i", "--idlimit", help="Set a custom idlimit.", type=float)
    parser.add_argument("-j", "--johnson", action='store_true', help="Use Stetson Johnson/Cousins filters and not SDSS")
    parser.add_argument("-k", "--makak", help="Makak tweaks.", action='store_true')
    parser.add_argument("-R", "--redlim", help="Do not get stars redder than this g-r.", type=float, default=5)
    parser.add_argument("-B", "--bluelim", help="Do not get stars bler than this g-r.", type=float, default=-5)
    parser.add_argument("-l", "--maglim", help="Do not get stars fainter than this limit.", type=float, default=17)
    parser.add_argument("-L", "--brightlim", help="Do not get any less than this mag from the catalog to compare.", type=float)
    parser.add_argument("-m", "--median", help="Give me just the median of zeropoints, no fitting.", action='store_true')
    parser.add_argument("-M", "--model", help="Read model from a file.", type=str)
    parser.add_argument("-n", "--nonlin", help="CCD is not linear, apply linear correction on mag.", action='store_true')
    parser.add_argument("-p", "--plot", help="Produce plots.", action='store_true')
    parser.add_argument("-r", "--reject", help="No outputs for Reduced Chi^2 > value.", type=float)
    parser.add_argument("-s", "--stars", action='store_true', help="Output fitted numbers to a file.")
    parser.add_argument("-S", "--sip", help="Order of SIP refinement for the astrometric solution (0=disable)", type=int)
    parser.add_argument("-t", "--fit-terms", help="Comma separated list of terms to fit", type=str)
    parser.add_argument("-T", "--trypar", type=str, help="Terms to examine to see if necessary (and include in the fit if they are).")
    parser.add_argument("-u", "--autoupdate", action='store_true', help="Update .det if .fits is newer", default=False)
    parser.add_argument("-U", "--terms", help="Terms to fit.", type=str)
    parser.add_argument("-v", "--verbose", action='store_true', help="Print debugging info.")
    parser.add_argument("-w", "--weight", action='store_true', help="Produce weight image.")
    parser.add_argument("-W", "--save-model", help="Write model into a file.", type=str)
    parser.add_argument("-x", "--fix-terms", help="Comma separated list of terms to keep fixed", type=str)
    parser.add_argument("-y", "--fit-xy", action='store_true', help="Fit xy tilt for each image separately (i.e. terms PX/PY)")
    parser.add_argument("-z", "--refit-zpn", action='store_true', help="Refit the ZPN radial terms.")
    parser.add_argument("-Z", "--szp", action='store_true', help="use SZP while fitting astrometry.")
    parser.add_argument("files", help="Frames to process", nargs='+', action='extend', type=str)
    opts = parser.parse_args(args)
    return opts

def airmass(z):
    """ Compute astronomical airmass according to Rozenberg(1966) """
    cz = np.cos(z)
    return 1/(cz + 0.025 * np.exp(-11*cz) )

def summag(magarray):
    """add two magnitudes of the same zeropoint"""
    return -2.5 * np.log10(np.sum( np.power(10.0, -0.4*np.array(magarray))) )

def get_atlas_dir(rasc, decl, width, height, directory, mlim):
    """get contents of one split of Atlas catalog (it is split into directories by magnitude)"""
    atlas_ecsv_tmp = f"atlas{os.getpid()}.ecsv"
    cmd = f"atlas {rasc} {decl} -rect {width},{height} -dir {directory} -mlim {mlim:.2f} -ecsv > {atlas_ecsv_tmp}"
    print(cmd)
    os.system(cmd)
    new = astropy.io.ascii.read(atlas_ecsv_tmp, format='ecsv')
    print(len(new))
    os.system("rm " + atlas_ecsv_tmp)
    return new

# filter transforms by Lupton (2005) (made for SDSS DR4)
# B: PC=-1.313 Bz=-0.2271 # With PanSTARRS has a serious trend and scatter
# V: PC=-0.4216 Vz=0.0038 # With PanSTARRS has a clear trend
# R: PD=0.2936 Rz=0.1439 # With PanSTARRS is ~ok (stars with r-i>2 tend off the line)
# I: PD=1.2444 Iz=0.3820 # With PanSTARRS is ok (more spread than PDPE, but no trends)

# filter transforms by mates (2024) PanSTARRS -> Stetson
# B: PC=-1.490989±0.008892 P2C=-0.125787±0.010429 P3C=0.022359±0.003590 Bz=-0.186304+/-0.003624
# V: PC=-0.510236±0.001648 Vz=0.0337082+/-0.004563
# R: PD=0.197420±0.005212 P2D=0.083113±0.004458 Rz=0.179943+/-0.002598
# I: PD=0.897087±0.012819 PE=0.575316±0.020487  Iz=0.423971+/-0.003196

def get_atlas(rasc, decl, width=0.25, height=0.25, mlim=17):
    """Load Atlas catalog from disk, (atlas command needs to be in path and working)"""
    cat = get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/00_m_16/', mlim)
    if mlim > 16: cat=astropy.table.vstack([cat, get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/16_m_17/', mlim)])
    if mlim > 17: cat=astropy.table.vstack([cat, get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/17_m_18/', mlim)])
    if mlim > 18: cat=astropy.table.vstack([cat, get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/18_m_19/', mlim)])
    if mlim > 19: cat=astropy.table.vstack([cat, get_atlas_dir(rasc, decl, width, height, '/home/mates/cat/atlas/19_m_20/', mlim)])

    # Fill in the Stetson/Johnson filter transformations
    gr = cat['Sloan_g'] - cat['Sloan_r']
    ri = cat['Sloan_r'] - cat['Sloan_i']
    iz = cat['Sloan_i'] - cat['Sloan_z']
    cat['Johnson_B'] = cat['Sloan_r'] + 1.490989 * gr + 0.125787 * gr * gr - 0.022359 * gr*gr*gr + 0.186304
    cat['Johnson_V'] = cat['Sloan_r'] + 0.510236 * gr - 0.0337082
    cat['Johnson_R'] = cat['Sloan_r'] - 0.197420 * ri - 0.083113 * ri * ri - 0.179943
    cat['Johnson_I'] = cat['Sloan_r'] - 0.897087 * ri - 0.575316 * iz - 0.423971

    cat.meta['astepoch']=2015.5
    return cat

# get another catalog from a file
def get_catalog(filename, mlim=17):
    """Expected use: read output of another image as a calibration source for this frame"""
    cat = astropy.table.Table()
    try:
        catalog = astropy.io.ascii.read(filename, format='ecsv')
        cat.add_column(astropy.table.Column(name='radeg', dtype=np.float64, data=catalog['ALPHA_J2000']))
        cat.add_column(astropy.table.Column(name='decdeg', dtype=np.float64, data=catalog['DELTA_J2000']))
        cat.add_column(astropy.table.Column(name='pmra', dtype=np.float64, data=catalog['_RAJ2000']*0))
        cat.add_column(astropy.table.Column(name='pmdec', dtype=np.float64, data=catalog['_DEJ2000']*0))
        # obviously, a single image output is single filter only
        fltnames=['Sloan_r','Sloan_i','Sloan_g','Sloan_z', 'Johnson_B','Johnson_V','Johnson_R','Johnson_I','J','c','o']
        for fltname in fltnames:
            cat.add_column(astropy.table.Column(name=fltname, dtype=np.float64, data=catalog['MAG_CALIB']))
        print(f"Succesfully read the ecsv catalog {filename} (mlim={mlim})")
        return cat
    except astropy.io.ascii.core.InconsistentTableError:
        pass
#        print("Hm... the catalog is not an ecsv!")
    try:
        catalog = astropy.io.ascii.read(filename, format='tab')
#        cat.add_column(catalog['_RAJ2000'])
        cat.add_column(astropy.table.Column(name='radeg', dtype=np.float64, data=catalog['_RAJ2000']))
        cat.add_column(astropy.table.Column(name='decdeg', dtype=np.float64, data=catalog['_DEJ2000']))
        cat.add_column(astropy.table.MaskedColumn(name='pmra', dtype=np.float64, data=catalog['pmRA']/1e6, fill_value=0))
        cat.add_column(astropy.table.MaskedColumn(name='pmdec', dtype=np.float64, data=catalog['pmDE']/1e6, fill_value=0))

        # we may do different arrangements of the primary/secondary filters
        # for BV@Tautenburg the logical choice is BP primary and BP-RP to correct. 
        # PC = (R-B), PD = (B-G), PE = (G-R), with -B Sloan_i, Gmag as a bas may be chosen, default is BPmag
        cat.add_column(astropy.table.MaskedColumn(name='Sloan_g', dtype=np.float64, data=catalog['RPmag'],fill_value=99.9))
        cat.add_column(astropy.table.MaskedColumn(name='Sloan_r', dtype=np.float64, data=catalog['BPmag'],fill_value=99.9))
        cat.add_column(astropy.table.MaskedColumn(name='Sloan_i', dtype=np.float64, data=catalog['Gmag'],fill_value=99.9))
        cat.add_column(astropy.table.MaskedColumn(name='Sloan_z', dtype=np.float64, data=catalog['RPmag'],fill_value=99.9))
        # BPmag values into the other filter columns 
        fltnames=['Johnson_B','Johnson_V','Johnson_R','Johnson_I','J','c','o']
        for fltname in fltnames:
            cat.add_column(astropy.table.MaskedColumn(name=fltname, dtype=np.float64, data=catalog['BPmag'], fill_value=99.9))
        print(f"Succesfully read the tab-sep-list catalog {filename} (mlim={mlim})")
        cat.meta['astepoch']=2015.5
        return cat[np.logical_not(np.any([
            np.ma.getmaskarray(cat['Sloan_g']),
            np.ma.getmaskarray(cat['Sloan_r']),
            np.ma.getmaskarray(cat['Sloan_i']),
            np.ma.getmaskarray(cat['pmra']),
            np.ma.getmaskarray(cat['pmdec']),
            cat['Sloan_r']>mlim,
            cat['Sloan_i']>mlim,
            cat['Sloan_g']>mlim,
            cat['Sloan_g']-cat['Sloan_r']<-1.5
            ] ,axis=0))]
    except astropy.io.ascii.core.InconsistentTableError:
        print("Catalog cannot be read as an ECSV or TAB-SEP-LIST file!")
    return None

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

#        if options.verbose:
#            print("Raw zeropoints: %s"%(np.array(x)-np.array(y)))
#            print("Median of this array is: Zo = %.3f"%(Zo))
        print("Filter: %s Zo: %f Zoe: %f"%(flt, Zo, Zoe))

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

def try_det(arg, verbose=False):
    """Try to open arg as an ecsv file, exit cleanly if it does not happen"""
    try:
        detfile = astropy.table.Table.read(arg, format="ascii.ecsv")
        if verbose: print(f"Argument {arg} is an ecsv table")
        return arg, detfile
    except (FileNotFoundError,OSError,UnicodeDecodeError,ValueError):
        pass
    try:
        detfile = astropy.table.Table.read(arg)
        if verbose: print(f"Argument {arg} is a table")
        return arg, detfile
    except (FileNotFoundError,OSError,UnicodeDecodeError,ValueError):
        if verbose: print(f"Argument {arg} is not a table")
        return None, None

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
    return basemag, fit_in_johnson

# ******** main() **********

def main():
    '''Take over the world.'''
    options = readOptions(sys.argv[1:])

    if options.verbose:
        print(f"{os.path.basename(sys.argv[0])} running in Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")
        print(f"Magnitude limit set to {options.maglim}")

    k = 0
    i = 0
    ast_projections = [ 'TAN', 'ZPN', 'ZEA', 'AZP' ]

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
    imgno=0

    for arg in options.files:

        detf, det = try_det(arg, options.verbose)
        if det is None: detf, det = try_det(os.path.splitext(arg)[0] + ".det", options.verbose)
        if det is None: detf, det = try_det(arg + ".det", options.verbose)

        fitsimgf, fitsimg = try_img(arg + ".fits", options.verbose)
        if fitsimg is None: fitsimgf, fitsimg = try_img(os.path.splitext(arg)[0] + ".fits", options.verbose)
        if fitsimg is None: fitsimgf, fitsimg = try_img(os.path.splitext(arg)[0], options.verbose)

        # with these, we should have the possible filenames covered, now lets see what we got
        # 1. have fitsimg, no .det: call cat2det, goto 3
        # 2. have fitsimg, have det, det is older than fitsimg: same as 1
        if (det is not None and fitsimg is not None and os.path.getmtime(fitsimgf) > os.path.getmtime(detf) and options.autoupdate) \
            or (det is None and fitsimg is not None):
            #cmd = f"cat2det.py {("","-v ")[options.verbose]}{fitsimgf}"
            cmd = f"cat2det.py -v {fitsimgf}"
            print(cmd)
            os.system(cmd)
            if options.verbose: print("Back in dophot")
            detf, det = try_det(os.path.splitext(fitsimgf)[0] + ".det", options.verbose)

        if det is None:
            print(f"I do not know what to do with {arg}")
            continue
        # print(type(det.meta)) # OrderedDict, so we can:
        with suppress(KeyError): det.meta.pop('comments')
        with suppress(KeyError): det.meta.pop('history')

        # 3. have fitsimg, have .det, note the fitsimg filename, close fitsimg and run
        # 4. have det, no fitsimg: just run, writing results into fits will be disabled
        if fitsimgf is not None: det.meta['filename'] = fitsimgf
        print(f"DetF={detf}, ImgF={fitsimgf}")

        some_file = open("det.reg", "w+")
        some_file.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=3"\
            " font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        for w_ra, w_dec in zip(det['X_IMAGE'],det['Y_IMAGE']):
            some_file.write(f"circle({w_ra:.7f},{w_dec:.7f},3\") # color=red width=3\n")
        some_file.close()

        imgwcs = astropy.wcs.WCS(det.meta)

        det['ALPHA_J2000'], det['DELTA_J2000'] = imgwcs.all_pix2world( [det['X_IMAGE']], [det['Y_IMAGE']], 1)

        if options.usewcs is not None:
            # we may import a WCS solution from a file:
            usewcs = astropy.table.Table.read(options.usewcs, format='ascii.ecsv')
            remove_junk(usewcs.meta)

            # we need to set the new center to what it should be:
            usectr = imgwcs.all_pix2world( [usewcs.meta['CRPIX1']], [usewcs.meta['CRPIX2']], 1)
            usewcs.meta['CRVAL1'] = usectr[0][0]
            usewcs.meta['CRVAL2'] = usectr[1][0]

            # and we need to rotate and rescale the transformation martix approprietely:
            dra = (usectr[0][0] - det.meta['CRVAL1']) * np.pi / 180.0

            scale = np.sqrt( \
                (np.power(usewcs.meta['CD1_1'], 2) + np.power(usewcs.meta['CD1_2'], 2) \
                + np.power(usewcs.meta['CD2_1'], 2) + np.power(usewcs.meta['CD2_2'], 2)) / 2) \
                / np.sqrt((np.power(det.meta['CD1_1'],2) + np.power(det.meta['CD1_2'], 2) \
                + np.power(det.meta['CD2_1'], 2) + np.power(det.meta['CD2_2'], 2)) / 2)

            print(f"Scale WCS by {scale} and rotate by {dra*180/np.pi}")

            usewcs.meta['CD1_1'] = det.meta['CD1_1'] * scale * np.cos(dra)
            usewcs.meta['CD1_2'] = det.meta['CD1_2'] * scale * -np.sin(dra)
            usewcs.meta['CD2_1'] = det.meta['CD2_1'] * scale * +np.sin(dra)
            usewcs.meta['CD2_2'] = det.meta['CD2_2'] * scale * np.cos(dra)

            #print(usewcs.meta)
            for term in [ "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2" ]:
                det.meta[term] = usewcs.meta[term]

            # the rest should be ok
            imgwcs = astropy.wcs.WCS(usewcs.meta)

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
        if options.verbose: print("EPOCH_DIFF=", epoch)
        cat['radeg'] += epoch*cat['pmra']
        cat['decdeg'] += epoch*cat['pmdec']

        if options.verbose: print(f"Catalog search took {time.time()-start:.3f}s")
        if options.verbose: print(f"Catalog contains {len(cat)} entries")


        some_file = open("cat.reg", "w+")
        some_file.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=3"\
            " font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
        some_file.write("box(%.7f,%.7f,%.3f\",%.3f\",%.3f) # color=red width=3\n"%(det.meta['CTRRA'],
            det.meta['CTRDEC'], 7200*det.meta['FIELD'], 7200*det.meta['FIELD'], 0))
        for w_ra, w_dec in zip(cat['radeg'],cat['decdeg']):
            some_file.write(f"circle({w_ra:.7f},{w_dec:.7f},3\") # color=red width=3\n")
        some_file.close()

        if options.idlimit: idlimit = options.idlimit
        else:
            try:
                idlimit = det.meta['FWHM']
                if options.verbose: print(f"idlimit set to fits header FWHM value of {idlimit} pixels.")
            except KeyError:
                idlimit = 2.0/3600
                if options.verbose: print(f"idlimit set to a hard-coded default of {idlimit} pixels.")

        # ===  identification with KDTree  ===
        Y = np.array([det['X_IMAGE'], det['Y_IMAGE']]).transpose()

        start = time.time()
        try:
            Xt = np.array(imgwcs.all_world2pix(cat['radeg'], cat['decdeg'],1))
        except (ValueError,KeyError):
            if options.verbose: print(f"Astrometry of {arg} sucks! Skipping.")
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
        if options.verbose: print(f"Object cross-id took {time.time()-start:.3f}s")

        some_file = open("noident.reg", "w+")
        some_file.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=3"\
            " font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
        for xx, dd in zip(nearest_ind, det):
            if len(xx) < 1: some_file.write("circle(%.7f,%.7f,%.3f\") # color=red width=3\n"%(dd["ALPHA_J2000"], dd["DELTA_J2000"], 18))
            else: some_file.write("circle(%.7f,%.7f,%.3f\")\n"%(dd["ALPHA_J2000"], dd["DELTA_J2000"],1.5*idlimit*dd.meta['PIXEL']))
        some_file.close()

        some_file = open("noidentXY.reg", "w+")
        some_file.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=3"\
            " font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        for xx, dd in zip(nearest_ind, det):
            if len(xx) < 1:
                if dd["MAGERR_AUTO"] < 1.091/5: some_file.write("circle(%.7f,%.7f,%.3f\") # color=red\n"%(\
                    dd["X_IMAGE"], dd["Y_IMAGE"],1.5*idlimit*dd.meta['PIXEL']))
                else: some_file.write("circle(%.7f,%.7f,%.3f\") # color=yellow\n"%(dd["X_IMAGE"], dd["Y_IMAGE"],1.5*idlimit*dd.meta['PIXEL']))
            else: some_file.write("circle(%.7f,%.7f,%.3f\")\n"%(dd["X_IMAGE"], dd["Y_IMAGE"],1.5*idlimit*dd.meta['PIXEL']))
        some_file.close()

        # non-id objects in frame:
        for xx, dd in zip(nearest_ind, det):
            if len(xx) < 1: tr.append(dd)

        # identify the target object
        target.append(None)
        if det.meta['OBJRA']<-99 or det.meta['OBJDEC']<-99:
            if options.verbose: print ("Target was not defined")
        else:
            if options.verbose: print (f"Target coordinates: {det.meta['OBJRA']:.6f} {det.meta['OBJDEC']:+6f}")
            Z = np.array(imgwcs.all_world2pix([det.meta['OBJRA']],[det.meta['OBJDEC']],1)).transpose()

            if np.isnan(Z[0][0]) or np.isnan(Z[0][1]):
                object_ind = None
                print ("Target transforms to Nan... :(")
            else:
                treeZ = KDTree(Z)
                object_ind, object_dist = treeZ.query_radius(Y, r=idlimit, return_distance=True, count_only=False)

                if object_ind is not None:
                    mindist = 2*idlimit # this is just a big number
                    for i, dd, d in zip(object_ind, object_dist, det):
                        if len(i) > 0 and dd[0] < mindist:
                            target[imgno] = d
                            mindist = dd[0]
                    if options.verbose:
                        if target[imgno] is None:
                            print ("Target was not found")
                        else:
                            print (f"Target is object id {target[imgno]['NUMBER']} at distance {mindist:.2f} px")

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
        for i, d in zip(nearest_ind, det):
            for k in i:

                magcat = cat[k][det.meta['REFILTER']]

                if det.meta['REJC']:
                    mag0 = cat[k]['Johnson_B']
                    mag1 = cat[k]['Johnson_V']
                    mag2 = cat[k]['Johnson_R']
                    mag3 = cat[k]['Johnson_I']
                    mag4 = cat[k]['J']
                else:
                    mag0 = cat[k]['Sloan_g']
                    mag1 = cat[k]['Sloan_r']
                    mag2 = cat[k]['Sloan_i']
                    mag3 = cat[k]['Sloan_z']
                    mag4 = cat[k]['J']

                if options.brightlim is not None and magcat < options.brightlim: continue
                if options.maglim is not None and magcat > options.maglim: continue
                if options.redlim is not None and (mag0 - mag2)/2 > options.redlim: continue
                if options.bluelim is not None and (mag0 - mag2)/2 < options.bluelim: continue

                x = np.append(x, magcat)
                y = np.append(y, np.float64(d['MAG_AUTO']))
                dy = np.append(dy, np.float64(d['MAGERR_AUTO']))

                tmpra, tmpdec = imgwcs.all_pix2world(d['X_IMAGE'], d['Y_IMAGE'], 1)
                rd = astropy.coordinates.SkyCoord( np.float64(tmpra), np.float64(tmpdec), unit=astropy.units.deg)
                rdalt = rd.transform_to(astropy.coordinates.AltAz(obstime=cas, location=loc))
                airm = airmass(np.pi/2-rdalt.alt.rad)

                aabs = np.append(aabs, airm)
                adif = np.append(adif, airm - det.meta['AIRMASS'])

                coord_x = np.append(coord_x, (np.float64(d['X_IMAGE'])-det.meta['CTRX'])/1024)
                coord_y = np.append(coord_y, (np.float64(d['Y_IMAGE'])-det.meta['CTRY'])/1024)
                image_x = np.append(image_x, (np.float64(d['X_IMAGE'])))
                image_y = np.append(image_y, (np.float64(d['Y_IMAGE'])))
                _dx = np.max((np.float64(d['ERRX2_IMAGE']),0.001))
                _dy = np.max((np.float64(d['ERRY2_IMAGE']),0.001))
                image_dxy = np.append(image_dxy, (np.sqrt(_dx*_dx+_dy*_dy)) )

                color1 = np.append(color1, np.float64(mag0-mag1)) #  g-r
                color2 = np.append(color2, np.float64(mag1-mag2)) #  r-i
                color3 = np.append(color3, np.float64(mag2-mag3)) #  i-z
                color4 = np.append(color4, np.float64(mag3-mag4)) #  z-J (J_{Vega} - J_{AB}  = 0.901)

                # catalog coordinates
                ra = np.append(ra, np.float64(cat[k]['radeg']))
                dec = np.append(dec, np.float64(cat[k]['decdeg']))
                img = np.append(img, np.int64(imgno))

        if len(dy)==0:
            print("Not enough stars to work with (Records:", len(dy),")")
            sys.exit(0)
        dy = np.sqrt(dy*dy+0.0004)
        # === fields are set up ===

        alldet = alldet+[det]
        imgno=imgno+1

    if imgno == 0:
        print("No images found to work with, quit")
        sys.exit(0)

    if len(y) == 0:
        print("No objects found to work with, quit")
        sys.exit(0)

    if options.verbose: print(f"Photometry will be fitted with {len(y)} objects from {imgno} files")
    # tady mame hvezdy ze snimku a hvezdy z katalogu nactene a identifikovane

    ffit = fotfit.fotfit(fit_xy=options.fit_xy)

    # Read a model to be fit from a file
    if options.model is not None:
        for modelfile in [  options.model,
                f"/home/mates/pyrt/model/{options.model}.mod",
                f"/home/mates/pyrt/model/{options.model}-{det.meta['FILTER']}.mod"]:
            try:
                print(f"Trying model {modelfile}")
                ffit.readmodel(modelfile)
                print(f"Model imported from {modelfile}")
                break
            except:
                print(f"Cannot open model {options.model}")

    ffit.fixall() # model read from the file is fixed even if it is not fixed in the file

    ffit.fixterm(["N1"], values=[0])

    if options.terms is not None:
        for term in options.terms.split(","):
            if term == "": continue # permit extra ","s
            if term[0] == '.': # not a real term: expansion script
                if term[1] == 'p':
                    pol_order = int(term[2:])
                    if options.verbose: print(f"set up a surface polynomial of order {pol_order:d}:")
                    polytxt = "P(x,y)="
                    for pp in range(1,pol_order+1):
                        if options.fit_xy and pp == 1:
                            continue
                        for rr in range(0,pp+1):
                            polytxt += f"+P{rr:d}X{pp-rr:d}Y*x**{rr:d}*y**{pp-rr:d}"
                            ffit.fitterm([f"P{rr:d}X{pp-rr:d}Y"], values=[1e-6])
                    if options.verbose: print(polytxt)
                if term[1] == 'r':
                    pol_order = int(term[2:])
                    if options.verbose: print(f"set up a polynomial of order {pol_order:d} in radius:")
                    polytxt = "P(r)="
                    for pp in range(1,pol_order+1):
                        polytxt += f"+P{pp:d}R*(x**2+y**2)**pp"
                        ffit.fitterm([f"P{pp:d}R"], values=[1e-6])
                    if options.verbose: print(polytxt)
            else:
                ffit.fitterm([term], values=[1e-6])

    imgno-=1

#    p0=[]
    #for term in fit_terms:
    #    p0+=[1e-6]
    #p0+=zeropoints
    ffit.zero = zeropoints
    #if options.fit_xy:
    #    p0 = np.append(p0, np.zeros(len(zeropoints)))
    #    p0 = np.append(p0, np.zeros(len(zeropoints)))

    start = time.time()

    fdata = (y, adif, coord_x, coord_y, color1, color2, color3, color4, img, x, dy)
    ffit.fit( fdata )
#    print(ffit.wssrndf)

    ok = ffit.residuals(ffit.fitvalues, fdata) < 5 * ffit.wssrndf
    fdata_ok = (y[ok], adif[ok], coord_x[ok], coord_y[ok], color1[ok], color2[ok], color3[ok], color4[ok], img[ok], x[ok], dy[ok])
    ffit.delin = True
    ffit.fit( fdata_ok )
#    print(ffit.wssrndf)

    ok = ffit.residuals(ffit.fitvalues, fdata) < 5 * ffit.wssrndf
    fdata_ok = (y[ok], adif[ok], coord_x[ok], coord_y[ok], color1[ok], color2[ok], color3[ok], color4[ok], img[ok], x[ok], dy[ok])
    ffit.delin = True
    ffit.fit( fdata_ok )

    print("Variance:", ffit.wssrndf)

    #ok = ffit.residuals(ffit.fitvalues, fdata) < 1.5
    #fdata_ok = (y[ok], adif[ok], coord_x[ok], coord_y[ok], color1[ok], color2[ok], color3[ok], color4[ok], img[ok], x[ok], dy[ok])
    #ffit.delin = True
    #ffit.fit( fdata_ok )

    # these objects are within 2-sigma of the photometric fit and snr > 10 sigma
    #ok = ffit.residuals(ffit.fitvalues, fdata) < 2.5
    #ok = np.all( [ffit.residuals0(ffit.fitvalues, fdata) < 2.0, dy < 1.091/10], axis=0)
    #print(len(y[ok]))

    if options.verbose: print("Fitting took %.3fs"%(time.time()-start))

    if options.verbose:
        print(ffit)

    if options.reject:
        if ffit.wssrndf > options.reject:
            if options.verbose:
                print("rejected (too large reduced chi2)")
            out_file = open("rejected", "a+")
            out_file.write("%s %.6f -\n"%(metadata[0]['FITSFILE'], ffit.wssrndf))
            out_file.close()
            sys.exit(0)

    if options.save_model is not None:
        ffit.savemodel(options.save_model)

    # """ REFIT ASTROMETRY """
    astra = 0 * dec
    astdec = 0 * ra
    ast = 0 * ra
    if options.astrometry:
        zpntest, ok2 = refit_astrometry(det, ra, dec, image_x, image_y, image_dxy, options)
        if zpntest is not None:
            # Update FITS file with new WCS
            fitsbase = os.path.splitext(arg)[0]
            newfits = fitsbase + "t.fits"
            if os.path.isfile(newfits):
                if options.verbose:
                    print("Will overwrite", newfits)
                os.unlink(newfits)
            os.system(f"cp {fitsbase}.fits {newfits}")
            zpntest.write(newfits)
            imgwcs = astropy.wcs.WCS(zpntest.wcs())
    else:
        ok2 = ok
    # ASTROMETRY END

    if options.stars:
        astra,astdec = imgwcs.all_world2pix( ra, dec, 1)
        stars_file  = open("stars", "a+")
        nn = 0
        stars_file.write("# x a image_x image_y color1 color2 color3 color4 y dy ra dec ok\n")
        mags = ffit.model( np.array(ffit.fitvalues), (y, adif, coord_x, coord_y, color1, color2, color3, color4, img, x, dy))

        while nn < len(x):
            stars_file.write("%8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %s %s\n"
                %(x[nn], aabs[nn], image_x[nn], image_y[nn], color1[nn], color2[nn], color3[nn], color4[nn], mags[nn], dy[nn], ra[nn], dec[nn], astra[nn], astdec[nn], ast[nn], ok[nn], ok2[nn]))
            nn = nn+1
        stars_file.close()


    zero, zerr = ffit.zero_val()

    for img in range(0, imgno+1):

    #    if options.verbose:
    #        print("astropy.io.fits.setval(%s,MAGZERO,0,value=%.3f)"%(det.meta['FITSFILE'],zero[img]))
    #    try:
    #        astropy.io.fits.setval(det.meta['FITSFILE'], "LIMMAG", 0, value=zero[img]+metadata[img]['LIMFLX3'])
    #        astropy.io.fits.setval(det.meta['FITSFILE'], "MAGZERO", 0, value=zero[img])
    #    except: print("  ... writing MAGZERO failed")
    #    if options.verbose:
    #        print("astropy.io.fits.setval(%s,RESPONSE,0,value=%s)"%(det.meta['FITSFILE'],ffit.oneline()))
    #    try: astropy.io.fits.setval(det.meta['FITSFILE'], "RESPONSE", 0, value=ffit.oneline())
    #    except: print("  ... writing RESPONSE failed")

        if options.astrometry:
            try:
                astropy.io.fits.setval( os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "LIMMAG", 0, value=zero[img]+metadata[img]['LIMFLX3'])
                astropy.io.fits.setval( os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "MAGZERO", 0, value=zero[img])
                astropy.io.fits.setval( os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "RESPONSE", 0, value=ffit.oneline())
            except: print("Writing LIMMAG/MAGZERO/RESPONSE to an astrometrized image failed")

        det = alldet[img]
    #    det.meta['comments']=[]
    #    det.meta['history']=[]
        #try:
        #    fn = os.path.splitext(det.meta['filename'])[0] + ".ecsv"
        #except KeyError:
        fn = os.path.splitext(detf)[0] + ".ecsv"
        det['MAG_CALIB'] = ffit.model(np.array(ffit.fitvalues), (det['MAG_AUTO'], metadata[img]['AIRMASS'], \
            det['X_IMAGE']/1024-metadata[img]['CTRX']/1024, det['Y_IMAGE']/1024-metadata[img]['CTRY']/1024,0,0,0,0, img,0,0)  )
        det['MAGERR_CALIB'] = np.sqrt(np.power(det['MAGERR_AUTO'],2)+np.power(zerr[img],2))

        det.meta['MAGZERO'] = zero[img]
        det.meta['DMAGZERO'] = zerr[img]
        det.meta['MAGLIMIT'] = metadata[img]['LIMFLX3']+zero[img]
        det.meta['WSSRNDF'] = ffit.wssrndf
        det.meta['RESPONSE'] = ffit.oneline()

        if (zerr[img] < 0.2) or (options.reject is None):
            det.write(fn, format="ascii.ecsv", overwrite=True)
        else:
            if options.verbose:
                print("rejected (too large uncertainity in zeropoint)")
            out_file = open("rejected", "a+")
            out_file.write(f"{metadata[0]['FITSFILE']} {ffit.wssrndf:.6f} {zerr[img]:.3f}\n")
            out_file.close()
            sys.exit(0)

        if options.flat or options.weight:
            print("generating the flat-field array")
            ffData = np.fromfunction(ffit.flat, [metadata[img]['IMGAXIS2'], metadata[img]['IMGAXIS1']] ,
                ctrx=metadata[img]['CTRX'], ctry=metadata[img]['CTRY'], img=img)
            # this would convert image to Jy if gain == 1
            if options.weight:
                if options.gain is None:
                    gain = 2.3
                else:
                    gain = options.gain

                # weight = signal^2 / (signal+noise)  (expected signal, measured noise)
                # expected signal for mag=m -> np.power(10,-0.4*(m - ffData)) (ffData is zeropoint)
                wwData = np.power(10,-0.4*(22-ffData)) # weighting to mag 22
                #wwData = np.power(wwData*gain,2) / (wwData*gain+78.5375*np.power(metadata[img]['BGSIGMA']*gain,2))
                wwData = np.power(wwData*gain,2) / (wwData*gain + 3.1315 * np.power(metadata[img]['BGSIGMA']*gain*metadata[img]['FWHM']/2,2))
                wwHDU = astropy.io.fits.PrimaryHDU(data=wwData)
                weightfile = os.path.splitext(det.meta['filename'])[0] + "w.fits"

                if os.path.isfile(weightfile):
                    print("Operation will overwrite an existing image: ", weightfile)
                    os.unlink(weightfile)

                wwFile = astropy.io.fits.open(weightfile, mode='append')

                wwFile.append(wwHDU)
                wwFile.close()

            if options.flat:
                ffData = np.power(10,0.4*(ffData-8.9))
                ffHDU = astropy.io.fits.PrimaryHDU(data=ffData)

                flatfile = os.path.splitext(det.meta['filename'])[0] + "f.fits"

                if os.path.isfile(flatfile):
                    print("Operation will overwrite an existing image: ", flatfile)
                    os.unlink(flatfile)

                ffFile = astropy.io.fits.open(flatfile, mode='append')

                ffFile.append(ffHDU)
                ffFile.close()

        tarid=0; tarid=0
        with suppress(KeyError): tarid=metadata[img]['TARGET']
        with suppress(KeyError): obsid=metadata[img]['OBSID']

        chartime = metadata[img]['JD'] + metadata[img]['EXPTIME'] / 2
        if options.date == 'char':
            with suppress(KeyError): chartime = metadata[img]['CHARTIME']
        if options.date == 'bjd':
            with suppress(KeyError): chartime = metadata[img]['BJD']

        if target[img] is not None:
            tarx=target[img]['X_IMAGE']/1024-metadata[img]['CTRX']/1024
            tary=target[img]['Y_IMAGE']/1024-metadata[img]['CTRY']/1024
#            tarr=tarx*tarx+tary*tary

            data_target = np.array([ [target[img]['MAG_AUTO']],[metadata[img]['AIRMASS']],[tarx],[tary],[0],[0],[0],[0],[img],[0],[0] ])
            mo = ffit.model( np.array(ffit.fitvalues), data_target)

            out_line="%s %14.6f %14.6f %s %3.0f %6.3f %4d %7.3f %6.3f %7.3f %6.3f %7.3f %6.3f %s %s ok"\
                %(metadata[img]['FITSFILE'], metadata[img]['JD'], chartime, flt, metadata[img]['EXPTIME'], metadata[img]['AIRMASS'], metadata[img]['IDNUM'], \
                zero[img], zerr[img], (metadata[img]['LIMFLX3']+zero[img]), ffit.wssrndf, \
                mo[0], target[img]['MAGERR_AUTO'], tarid, obsid)

        else:
            out_line="%s %14.6f %14.6f %s %3.0f %6.3f %4d %7.3f %6.3f %7.3f %6.3f  99.999  99.999 %s %s not_found"\
                %(metadata[img]['FITSFILE'], metadata[img]['JD'], chartime, flt, metadata[img]['EXPTIME'], metadata[img]['AIRMASS'], metadata[img]['IDNUM'], \
                zero[img], zerr[img], (metadata[img]['LIMFLX3']+zero[img]), ffit.wssrndf, tarid, obsid)

        print(out_line)

        # dophot.dat for the result line
        try:
            out_file = open("dophot.dat", "a")
            out_file.write(out_line+"\n")
            out_file.close()
        except:
            pass

# this way, the variables local to main() are not globally available, avoiding some programming errors
if __name__ == "__main__":
    main()
