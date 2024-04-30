#!/usr/bin/env python3

# awk '{for(i=0;i<$1;i++)for(j=0;j<$1;j++){if(i>0 || j>0)printf("P%dX%dY,", i, j);}}'

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
#import matplotlib.pyplot as plt
#import scipy
import numpy as np
import argparse

import scipy.optimize as fit
from sklearn.neighbors import KDTree

import zpnfit
import fotfit

call = "debile" # call me this way
fix25 = True # whether to fix pogson at 2.5 when fitting mag limit

def exportColumnsForDS9(columns, file="ds9.reg", size=10, width=3, color="red"):
    some_file = open(file, "w+")
    some_file.write("# Region file format: DS9 version 4.1\nglobal color=%s dashlist=8 3 width=%d font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n"%(color, width))
    for aa, dd in zip(columns[0], columns[1]):
        some_file.write("circle(%.7f,%.7f,%.3f\") # color=%s width=%d\n"%(aa, dd, size, color, width))

def readOptions(args=sys.argv[1:]):
  parser = argparse.ArgumentParser(description="Compute photometric calibration for a FITS image.")
  parser.add_argument("-a", "--astrometry", help="Refit astrometric solution using photometry-selected stars", action='store_true')
  parser.add_argument("-A", "--aterms", help="Terms to fit for astrometry.", type=str)
  parser.add_argument("-b", "--usewcs", help="Use this astrometric solution (file with header)", type=str)
  parser.add_argument("-c", "--catalog", action='store', help="Use this catalog as a reference.")
  parser.add_argument("-e", "--enlarge", help="Enlarge catalog search region", type=float)
  parser.add_argument("-f", "--filter", help="Override filter info from fits", type=str)
  parser.add_argument("-F", "--flat", help="Produce flats.", action='store_true')
  parser.add_argument("-g", "--tryflt", action='store_true', help="Try different filters.")
  parser.add_argument("-G", "--gain", action='store', help="Provide camera gain.", type=float)
  parser.add_argument("-i", "--idlimit", help="Set a custom idlimit.", type=float)
  parser.add_argument("-k", "--makak", help="Makak tweaks.", action='store_true')
  parser.add_argument("-l", "--maglim", help="Do not get any more than this mag from the catalog to compare.", type=float)
  parser.add_argument("-L", "--brightlim", help="Do not get any less than this mag from the catalog to compare.", type=float)
  parser.add_argument("-m", "--median", help="Give me just the median of zeropoints, no fitting.", action='store_true')
  parser.add_argument("-M", "--model", help="Read model from a file.", type=str)
  parser.add_argument("-n", "--nonlin", help="CCD is not linear, apply linear correction on mag.", action='store_true')
  parser.add_argument("-p", "--plot", help="Produce plots.", action='store_true')
  parser.add_argument("-r", "--reject", help="No outputs for Reduced Chi^2 > value.", type=float)
  parser.add_argument("-s", "--stars", action='store_true', help="Output fitted numbers to a file.")
  parser.add_argument("-t", "--fit-terms", help="Comma separated list of terms to fit", type=str)
  parser.add_argument("-T", "--trypar", help="Terms to examine to see if necessary (and include in the fit if they are).", type=str)
  parser.add_argument("-U", "--terms", help="Terms to fit.", type=str)
  parser.add_argument("-v", "--verbose", action='store_true', help="Print debugging info.")
  parser.add_argument("-w", "--weight", action='store_true', help="Produce weight image.")
  parser.add_argument("-W", "--save-model", help="Write model into a file.", type=str)
  parser.add_argument("-x", "--fix-terms", help="Comma separated list of terms to keep fixed", type=str)
  parser.add_argument("-y", "--fit-xy", help="Fit xy tilt for each image separately (i.e. terms PX/PY)", action='store_true')
  parser.add_argument("-z", "--refit-zpn", action='store_true', help="Refit the ZPN radial terms.")
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

# fix this...
##    exportColumnsForDS9(cc["radeg"], cc["decdeg"], file="atlas.reg", size=10.0)
#   some_file = open("atlas.reg", "w+")
#   some_file.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
#   for cc in cat:
#       some_file.write("circle(%.7f,%.7f,%.3f\")\n"%(cc["radeg"], cc["decdeg"],10.0))
#   some_file.close()

    return cat

# get another catalog from a file
def get_catalog(filename):

    cat = astropy.table.Table() #name='Atlas catalog query')

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

def check_filter(nearest_ind, det):
    ''' check filter and return the most probable filter '''

    bestflt="-"
    besterr=np.float64(9e99)
    bestzero=0

    for flt in filters:
        x = []
        y = []
        dy = []

        for i, d in zip(nearest_ind, det):

            mag1 = np.float64(9e99)
            mag2 = np.float64(9e99)

            for k in i:
                mag1 = summag(mag1, cat[k][first_filter[flt]])
                mag2 = summag(mag2, cat[k][second_filter[flt]])

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

    return(bestflt)

def median_zeropoint(nearest_ind, det, flt, forceAB=False):
    ''' check filter and return the most probable filter '''

    bestflt="-"
    besterr=np.float64(9e99)
    bestzero=0

    x = []
    y = []
    dy = []

    for i, d in zip(nearest_ind, det):

        mag1 = np.float64(9e99)
        mag2 = np.float64(9e99)

        try:
            test=color_term0[flt]
        except KeyError:
            flt='R'

        if forceAB: zero=0
        else:
            zero=color_term0[flt]



        for k in i:
            mag1 = summag(mag1, cat[k][first_filter[flt]])
            mag2 = summag(mag2, cat[k][second_filter[flt]])

        if len(i)>0:
            x = np.append(x, mag1 + color_termB[flt] * (mag2 - mag1))
            y = np.append(y, np.float64(d['MAG_AUTO']))
            dy = np.append(dy, np.float64(d['MAGERR_AUTO']))

    # median would complain but not crash if there are no identifications, this is a clean way out
    if len(x)<=0:
        # print("No identified stars within image");
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
        det.meta['FITSFILE'], det.meta['JD'], det.meta['JD_END'], flt, det.meta['EXPTIME'],
        det.meta['AIRMASS'], idnum, Zo, Zoe, (det.meta['LIMFLX10']+Zo),
        (det.meta['LIMFLX3']+Zo), tarmag, tarerr, 0, det.meta['OBSID'], tarstatus))

    return

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
    options.maglim = 17

if options.verbose:
    print("%s running in python %d.%d.%d"%(os.path.basename(sys.argv[0]), sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print("Magnitude limit set to %.2f"%(options.maglim))

if options.plot:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Unable to load matplotlib.pyplot, disabling plots")
        options.plot = False

dat = astropy.table.Table()
dat.add_column(astropy.table.Column(name='ra', dtype=np.float64, data=[]))
dat.add_column(astropy.table.Column(name='dec', dtype=np.float64, data=[]))
dat.add_column(astropy.table.Column(name='mag_cat', dtype=np.float64, data=[]))
dat.add_column(astropy.table.Column(name='mag_cat2', dtype=np.float64, data=[]))
dat.add_column(astropy.table.Column(name='airmass', dtype=np.float64, data=[]))
dat.add_column(astropy.table.Column(name='img', dtype=np.int32, data=[]))
dat.add_column(astropy.table.Column(name='time', dtype=np.int32, data=[]))
dat.add_column(astropy.table.Column(name='x', dtype=np.float64, data=[]))
dat.add_column(astropy.table.Column(name='y', dtype=np.float64, data=[]))
dat.add_column(astropy.table.Column(name='id', dtype=np.int32, data=[]))

filters = ['B', 'V', 'R', 'I', 'N', 'Sloan_r', 'Sloan_g', 'Sloan_i', 'Sloan_z']

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

l = 0
k = 0
u = []
w = 0
i = 0
v = 0
rV = {}
nothing = 0

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

imgno=0

for arg in options.files:

    try:
        det = astropy.table.Table.read(arg)
        det.meta['filename'] = arg;
        if options.verbose: print("Input file:", arg)
    except:
        det = None
        try:
            fn = os.path.splitext(arg)[0] + ".det"

            if not os.path.isfile(fn):
                try:
                    print("%s does not seem to exist, will run cat2det.py to make it"%(fn));
                    os.system("cat2det.py %s"%(arg))
                except:
                    print("Failed to run cat2det.py");

            if det is None:
                try:
                    det = astropy.table.Table.read(fn)
                except:
                    det = None

            if det is None:
                try:
                    det = astropy.table.Table.read(fn, format="ascii.ecsv")
                except:
                    det = None

            det.meta['filename'] = fn;
            if options.verbose: print("Input file:", fn)
        except:
                if options.verbose: print("File %s sucks! Skipping."%(arg))
                continue

    #exportColumnsForDS9([det["ALPHA_J2000"], det["DELTA_J2000"]], file="detections.reg", size=8.0, color="yellow")

#    print(type(det.meta))

    remove_junk(det.meta)

    imgwcs = astropy.wcs.WCS(det.meta)
    det['ALPHA_J2000'], det['DELTA_J2000'] = imgwcs.all_pix2world( [det['X_IMAGE']], [det['Y_IMAGE']], 1)
    #det['ALPHA_J2000']= det['ALPHA_J2000'] - 1.0;
    #det['DELTA_J2000']= det['DELTA_J2000'] - 1.0;

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

        print("Scale WCS by %f and rotate by %f"%(scale, dra*180/np.pi))

        usewcs.meta['CD1_1'] = det.meta['CD1_1'] * scale * np.cos(dra)
        usewcs.meta['CD1_2'] = det.meta['CD1_2'] * scale * -np.sin(dra)
        usewcs.meta['CD2_1'] = det.meta['CD2_1'] * scale * +np.sin(dra)
        usewcs.meta['CD2_2'] = det.meta['CD2_2'] * scale * np.cos(dra)

        #print(usewcs.meta)       
        for term in [ "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2" ]:
            det.meta[term] = usewcs.meta[term]
 
        # the rest should be ok
        imgwcs = astropy.wcs.WCS(usewcs.meta)

        #exit(0)

    try:
        field=det.meta['FIELD']
    except:
        det.meta['FIELD'] = 180

    if options.enlarge is not None:
        enlarge = options.enlarge
    else: enlarge=1

    start = time.time()
    if options.makak:
        cat = astropy.table.Table.read('/home/mates/test/catalog.fits')
        ctr = SkyCoord(det.meta['CTRRA']*astropy.units.deg,det.meta['CTRDEC']*astropy.units.deg,frame='fk5')
        c2 = SkyCoord(cat['radeg']*astropy.units.deg,cat['decdeg']*astropy.units.deg,frame='fk5').separation(ctr) < det.meta['FIELD']*astropy.units.deg / 2
        cat=cat[c2]
    else:
        if options.catalog: 
            cat = get_catalog(options.catalog)
        else: 
            cat = get_atlas(det.meta['CTRRA'], det.meta['CTRDEC'], width=enlarge*det.meta['FIELD'], height=enlarge*det.meta['FIELD'], mlim=options.maglim)
            epoch = ( det.meta['JD'] - 2457204.5 ) / 365.2425 # Epoch for PM correction (Gaia DR2@2015.5)
            cat['radeg'] += epoch*cat['pmra']
            cat['decdeg'] += epoch*cat['pmdec']
    if options.verbose: print("Catalog search took %.3fs"%(time.time()-start))
    if options.verbose: print("Catalog contains %d entries"%(len(cat)))
    

    some_file = open("cat.reg", "w+")
    some_file.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
    some_file.write("box(%.7f,%.7f,%.3f\",%.3f\",%.3f) # color=red width=3\n"%(det.meta['CTRRA'], det.meta['CTRDEC'], 7200*det.meta['FIELD'], 7200*det.meta['FIELD'], 0))
    for w_ra, w_dec in zip(cat['radeg'],cat['decdeg']):
        some_file.write("circle(%.7f,%.7f,%.3f\") # color=red width=3\n"%(w_ra, w_dec, 3))
    some_file.close()

    if options.idlimit: idlimit = options.idlimit
    else: 
        try:
            idlimit = det.meta['FWHM']
            if options.verbose: print("idlimit set to fits header FWHM value of %f pixels."%(idlimit))
        except:
            idlimit = 2.0/3600
            if options.verbose: print("idlimit set to a hard-coded default of %f pixels."%(idlimit))

    # ===  identification with KDTree  ===
    Y = np.array([det['X_IMAGE'], det['Y_IMAGE']]).transpose()

    start = time.time()
    try:
        Xt = np.array(imgwcs.all_world2pix(cat['radeg'], cat['decdeg'],1))
        # careful: Xselect conditions are inverted, i.e. what should not be left in
#        Xselect = np.any([np.isnan(Xt[0]),np.isnan(Xt[1]),Xt[0]<0,Xt[1]<0,Xt[0]>det.meta['IMAGEW'],Xt[1]>det.meta['IMAGEH'],cat['Sloan_r']>6.0], axis=0)
#        X = Xt.transpose()[~Xselect]

        X = Xt.transpose()
    except:
        if options.verbose: print("Astrometry of %s sucks! Skipping."%(arg))
        continue
    tree = KDTree(X)
    nearest_ind, nearest_dist = tree.query_radius(Y, r=idlimit, return_distance=True, count_only=False)
    if options.verbose: print("Object cross-id took %.3fs"%(time.time()-start))

    some_file = open("noident.reg", "w+")
    some_file.write("# Region file format: DS9 version 4.1\nglobal color=red dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
    for xx, dd in zip(nearest_ind, det):
        if len(xx) < 1: some_file.write("circle(%.7f,%.7f,%.3f\") # color=red width=3\n"%(dd["ALPHA_J2000"], dd["DELTA_J2000"], 18))
        else: some_file.write("circle(%.7f,%.7f,%.3f\")\n"%(dd["ALPHA_J2000"], dd["DELTA_J2000"],1.5*idlimit*dd.meta['PIXEL']))
    some_file.close()

    some_file = open("noidentXY.reg", "w+")
    some_file.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
    for xx, dd in zip(nearest_ind, det):
        if len(xx) < 1: 
            if dd["MAGERR_AUTO"] < 1.091/5:some_file.write("circle(%.7f,%.7f,%.3f\") # color=red\n"%(dd["X_IMAGE"], dd["Y_IMAGE"],1.5*idlimit*dd.meta['PIXEL']))
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
        if options.verbose: print ("Target coordinates: %.6f %+.6f"%(det.meta['OBJRA'], det.meta['OBJDEC']))
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
                        print ("Target is object id %s at distance %.2f px"%(target[imgno]['NUMBER'], mindist))

    # === objects identified ===

    # TODO: handle discrepancy between estimated and fitsheader filter
    flt = det.meta['FILTER']

    if options.tryflt:
        fltx = check_filter(nearest_ind, det)
#        print("Filter: %s"%(fltx))
        flt = fltx

    # crude estimation of an image photometric zeropoint
    Zo, Zoe, Zn = median_zeropoint(nearest_ind, det, flt)

    if Zn == 0:
        print("Warning: no identified stars in %s, skip image"%(det.meta['FITSFILE']))
        continue

    if options.tryflt:
        print_image_line(det, flt, Zo, Zoe, target[imgno], Zn)
        exit(0)

    # === fill up fields to be fitted ===

#   zeropoints.append(Zo)
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

    # make pairs to be fitted
    for i, d in zip(nearest_ind, det):
        """
        mag0 = np.float64(9e99)
        mag1 = np.float64(9e99)
        mag2 = np.float64(9e99)
        mag3 = np.float64(9e99)
        mag4 = np.float64(9e99)

        for k in i:
            mag0 = summag(mag0, cat[k]['Sloan_g'])
            mag1 = summag(mag1, cat[k]['Sloan_r'])
            mag2 = summag(mag2, cat[k]['Sloan_i'])
            mag3 = summag(mag3, cat[k]['Sloan_z'])
            mag4 = summag(mag4, cat[k]['J'])
        """
        for k in i:
            mag0 = cat[k]['Sloan_g']
            mag1 = cat[k]['Sloan_r']
            mag2 = cat[k]['Sloan_i']
            mag3 = cat[k]['Sloan_z']
            mag4 = cat[k]['J']

            magcat = mag2 # 1 + zero + color_termB[flt] * (mag2 - mag1)
            magdet = Zo + np.float64(d['MAG_AUTO'])
            errdet = np.float64(d['MAGERR_AUTO'])

            if options.brightlim is not None and magcat < options.brightlim:
#                print("* Star id.%d is too bright in the catalogue"%(d['NUMBER']))
                continue

#            if np.abs((magcat - magdet)/np.sqrt(errdet*errdet+Zoe*Zoe)) > 5.0:
#                print("* Star id.%d is %.3f mag brighter than in catalogue"%(d['NUMBER'], magcat - magdet))
#            else:
#            m = np.append(m, mag1 + zero + color_termB[flt] * (mag2 - mag1))
            x = np.append(x, mag1)
            y = np.append(y, np.float64(d['MAG_AUTO']))
            dy = np.append(dy, np.float64(d['MAGERR_AUTO']))

            # each object has its own airmass (not the image center)
#            a = np.append(a, np.float64(det.meta['AIRMASS']))
            tmpra, tmpdec = imgwcs.all_pix2world(d['X_IMAGE'], d['Y_IMAGE'], 1)
            rd = astropy.coordinates.SkyCoord( np.float64(tmpra), np.float64(tmpdec), unit=astropy.units.deg)
            rdalt = rd.transform_to(astropy.coordinates.AltAz(obstime=cas, location=loc))
            airm = airmass(np.pi/2-rdalt.alt.rad)

#            print("Airmass = %.4f"%(airmass(rdalt.alt.rad)))

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
            # k = i[0]
            ra = np.append(ra, np.float64(cat[k]['radeg']))
            dec = np.append(dec, np.float64(cat[k]['decdeg']))
            img = np.append(img, np.int64(imgno))

    if len(dy)==0:
        print("Not enough stars to work with (Records:", len(dy),")")
        exit(0)
    dy = np.sqrt(dy*dy+0.0004)
    scnt = len(x)
    # === fields are set up ===

    alldet = alldet+[det]
    imgno=imgno+1

    # tady mame hvezdy ze snimku a hvezdy z katalogu nactene a identifikovane

ffit = fotfit.fotfit(fit_xy=options.fit_xy) 
    
# Read a model to be fit from a file
if options.model is not None:
    modelfile_list = [  options.model,
                        "/home/mates/pyrt/model/%s.mod"%(options.model),
                        "/home/mates/pyrt/model/%s-%s.mod"%(options.model, det.meta['FILTER'])]
    for modelfile in modelfile_list:
        try:
            print("Trying model %s"%(modelfile))
            ffit.readmodel(modelfile)
            print("Model imported from %s"%(modelfile))
            break
        except:
            print("Cannot open model %s"%(options.model))

ffit.fixall() # model read from the file is fixed even if it is not fixed in the file

ffit.fixterm(["N1"], values=[0])

if options.terms is not None:
    for term in options.terms.split(","):
        if term[0] == '.': # not a real term: expansion script
            if term[1] == 'p':
                pol_order = int(term[2:])
                print(f"set up a polynimial of {pol_order:d} order:")
                print("P(x,y)=\\")
                for pp in range(1,pol_order+1):
                    if options.fit_xy and pp == 1:
                        continue
                    for rr in range(0,pp+1):
                        print(f"+P{rr:d}X{pp-rr:d}Y*x**{rr:d}*y**{pp-rr:d}\\")
                        ffit.fitterm(["P%dX%dY"%(rr,pp-rr)], values=[1e-6])
        else:
            ffit.fitterm([term], values=[1e-6])

imgno-=1

p0=[]
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
print(ffit.wssrndf)

ok = ffit.residuals(ffit.fitvalues, fdata) < 5 * ffit.wssrndf
fdata_ok = (y[ok], adif[ok], coord_x[ok], coord_y[ok], color1[ok], color2[ok], color3[ok], color4[ok], img[ok], x[ok], dy[ok]) 
ffit.delin = True 
ffit.fit( fdata_ok )
print(ffit.wssrndf)

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
#    print(ffit.oneline())

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

    try:
        camera = det.meta['CCD_NAME']
    except:
        camera = "C0"

    adata_ok=(image_x[ok], image_y[ok], ra[ok], dec[ok], image_dxy[ok])

# C3@sbt is ok with TAN
    if camera == "C1" or camera == "C2" or camera == "makak" or camera == "makak2" or camera == "NF4" or camera == "ASM1":
        zpntest = zpnfit.zpnfit(proj="ZPN")  # TAN/ZPN/ZEA/AZP/SZP
        zpntest.fixterm(["PV2_1"], [1])
    else:
        zpntest = zpnfit.zpnfit(proj="TAN")  # TAN/ZPN/ZEA/AZP/SZP

    keys_invalid=False
    for term in [ "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2" ]:
        try:
            zpntest.fitterm([term], [det.meta[term]])
        except KeyError:
            keys_invalid=True

    if keys_invalid: 
        try: # one more thing to try is to interpret the old-fashioned WCS with CROTA
            print(det.meta['CDELT1'],det.meta['CROTA1'],det.meta['CDELT2'],det.meta['CROTA2'])
            zpntest.fitterm(['CD1_1'], [det.meta['CDELT1'] * np.cos(det.meta['CROTA1']*np.pi/180)])
            zpntest.fitterm(['CD1_2'], [det.meta['CDELT1'] * np.sin(det.meta['CROTA1']*np.pi/180)])
            zpntest.fitterm(['CD2_1'], [det.meta['CDELT2'] * -np.sin(det.meta['CROTA2']*np.pi/180)])
            zpntest.fitterm(['CD2_2'], [det.meta['CDELT2'] * np.cos(det.meta['CROTA2']*np.pi/180)])
            keys_invalid=False
        except KeyError:
            keys_invalid=True

    print(zpntest)
    if keys_invalid: 
        ok2 = ok
        print("I do not understand the WCS to be fitted, next...")
    else:

        if camera == "C1":
            if options.refit_zpn: zpntest.fitterm(["PV2_3", "PV2_5"], [7.5, 386.1])
            else: zpntest.fixterm(["PV2_3", "PV2_5"], [7.5, 386.1])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [2100,2100])
        if camera == "C2":
            if options.refit_zpn: zpntest.fitterm(["PV2_3", "PV2_5"], [8.255, 343.8])
            else: zpntest.fixterm(["PV2_3", "PV2_5"], [8.255, 343.8])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [2065.0,2035.0])
        if camera == "makak2" or camera == "makak":
            if options.refit_zpn: 
                zpntest.fitterm(["PV2_3", "PV2_5"], [0.131823, 0.282538])
                zpntest.fitterm(["CRPIX1", "CRPIX2"], [813.6,622.8])
            else: 
                zpntest.fixterm(["PV2_3", "PV2_5"], [0.131823, 0.282538])
                zpntest.fixterm(["CRPIX1", "CRPIX2"], [813.6,622.8])
        if camera == "NF4":
            zpntest.fitterm(["PV2_3"], [65.913305900171])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [522.75,569.96])
        if camera == "ASM1":
            if options.refit_zpn:
                zpntest.fitterm(["PV2_3", "PV2_5", "PV2_5"], [-0.0388566,0.001255,-0.002769])
                zpntest.fitterm(["CRPIX1", "CRPIX2"], [2054.5,2059.0])
            else:
                zpntest.fixterm(["PV2_3", "PV2_5", "PV2_5"], [-0.0388566,0.001255,-0.002769])
                zpntest.fixterm(["CRPIX1", "CRPIX2"], [2054.5,2059.0])
    #    if camera == "makak":
    #        zpntest.fitterm(["PV2_3", "PV2_5"], [0.131823, 0.282538])
    #        zpntest.fitterm(["CRPIX1", "CRPIX2"], [813.6,622.8])
            
        #zpntest.fixterm(["CRPIX1", "CRPIX2"],[det.meta['IMAGEW']/2, det.meta['IMAGEH']/2])

        zpntest.fit(adata_ok)
        print(zpntest.sigma, zpntest.wssrndf)
        ok2 = np.all([ok, 
                zpntest.residuals(zpntest.fitvalues, (image_x, image_y, ra, dec, image_dxy))
                < 3*zpntest.wssrndf], axis=0)
                #< (3*zpntest.sigma)], axis=0)
        adata_ok = (image_x[ok2], image_y[ok2], ra[ok2], dec[ok2], image_dxy[ok2])
        zpntest.delin = True 
        zpntest.fit(adata_ok)
        print(zpntest.sigma, zpntest.wssrndf)
        ok2 = np.all([ok, 
                zpntest.residuals(zpntest.fitvalues, (image_x, image_y, ra, dec, image_dxy))
                < 3.0*zpntest.wssrndf ], axis=0)
                #< (2.0*zpntest.sigma)], axis=0)
        adata_ok = (image_x[ok2], image_y[ok2], ra[ok2], dec[ok2], image_dxy[ok2])
        zpntest.delin = True 
        zpntest.fit(adata_ok)
        print(zpntest.sigma, zpntest.wssrndf)

        zpntest.savemodel("astmodel.ecsv")
        print(zpntest)
        #zpn_new=zpnfit.zpnfit(file="astmodel.ecsv")
        #print(zpn_new)

        fitsbase = os.path.splitext(arg)[0]
        newfits = fitsbase + "t.fits"
        
        if os.path.isfile(newfits):
            if options.verbose:
                print("Will overwrite", newfits)
            os.unlink(newfits)

        os.system("cp %s.fits %s"%(fitsbase, newfits))

        # Write a new fits with the calculated WCS solution
        zpntest.write(newfits)

        # reinit the WCS structure for us here
        imgwcs = astropy.wcs.WCS(zpntest.wcs())

    # """ END ASTROMETRY REFIT """

if not options.astrometry:
    ok2 = ok

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

    if options.verbose:
        print("astropy.io.fits.setval(%s,MAGZERO,0,value=%.3f)"%(det.meta['FITSFILE'],zero[img]))
    try: 
        astropy.io.fits.setval(det.meta['FITSFILE'], "LIMMAG", 0, value=zero[img]+metadata[img]['LIMFLX3'])
        astropy.io.fits.setval(det.meta['FITSFILE'], "MAGZERO", 0, value=zero[img])
    except: print("  ... writing MAGZERO failed")
    if options.verbose:
        print("astropy.io.fits.setval(%s,RESPONSE,0,value=%s)"%(det.meta['FITSFILE'],ffit.oneline()))
    try: astropy.io.fits.setval(det.meta['FITSFILE'], "RESPONSE", 0, value=ffit.oneline())
    except: print("  ... writing RESPONSE failed")

    if options.astrometry:
        try: 
            astropy.io.fits.setval( os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "LIMMAG", 0, value=zero[img]+metadata[img]['LIMFLX3'])
            astropy.io.fits.setval( os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "MAGZERO", 0, value=zero[img])
        except: print("  ... writing MAGZERO to astrometrized image failed")
        try: astropy.io.fits.setval( os.path.splitext(det.meta['FITSFILE'])[0]+"t.fits", "RESPONSE", 0, value=ffit.oneline())
        except: print("  ... writing RESPONSE to astrometrized image failed")

    det = alldet[img]
    det.meta['comments']=[]
    det.meta['history']=[]
    fn = os.path.splitext(det.meta['filename'])[0] + ".ecsv"
    det['MAG_CALIB'] = ffit.model(np.array(ffit.fitvalues), (det['MAG_AUTO'], metadata[img]['AIRMASS'], det['X_IMAGE']/1024-metadata[img]['CTRX']/1024, det['Y_IMAGE']/1024-metadata[img]['CTRY']/1024,0,0,0,0, img,0,0)  )
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
            print("rejected")
        out_file = open("rejected (too large uncertainity in zeropoint)", "a+")
        out_file.write("%s %.6f %.3f\n"%(metadata[0]['FITSFILE'], ffit.wssrndf, zerr[img]))
        out_file.close()
        sys.exit(0)

    if options.flat or options.weight:
        print("generating the flat-field array")
        ffData = np.fromfunction(ffit.flat, [metadata[img]['IMGAXIS2'], metadata[img]['IMGAXIS1']] , ctrx=metadata[img]['CTRX'], ctry=metadata[img]['CTRY'], img=img)
        # this would convert image to Jy if gain == 1
        if options.weight:
            if options.gain is None:
                g = 2.3
            else:
                g = options.gain
            wwData = np.power(10,0.4*(ffData-8.9)) # weighting to mag 12
            wwData = np.power(wwData*g,2) / (wwData*g+78.5375*np.power(metadata[img]['BGSIGMA']*g,2))
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

    try:
        tarid=metadata[img]['TARGET']
    except:
        tarid=0
    try:
        obsid=metadata[img]['OBSID']
    except:
        obsid=0

    try:
        chartime = metadata[img]['CHARTIME']
    except:
        chartime = metadata[img]['JD'] + metadata[img]['EXPTIME'] / 2

    if target[img] != None:
        tarx=target[img]['X_IMAGE']/1024-metadata[img]['CTRX']/1024
        tary=target[img]['Y_IMAGE']/1024-metadata[img]['CTRY']/1024
        tarr=tarx*tarx+tary*tary

        data_target = np.array([ [target[img]['MAG_AUTO']],[metadata[img]['AIRMASS']],[tarx],[tary],[0],[0],[0],[0],[img],[0],[0] ])
        mo = ffit.model( np.array(ffit.fitvalues), data_target)


        print("%s %14.6f %14.6f %s %3.0f %6.3f %4d %7.3f %6.3f %7.3f %6.3f %7.3f %6.3f %s %s ok"\
            %(metadata[img]['FITSFILE'], metadata[img]['JD'], chartime, flt, metadata[img]['EXPTIME'], metadata[img]['AIRMASS'], metadata[img]['IDNUM'], \
            zero[img], zerr[img], (metadata[img]['LIMFLX3']+zero[img]), ffit.wssrndf, \
            mo[0], target[img]['MAGERR_AUTO'], tarid, obsid))

    else:
        print("%s %14.6f %14.6f %s %3.0f %6.3f %4d %7.3f %6.3f %7.3f %6.3f  99.999  99.999 %s %s not_found"\
            %(metadata[img]['FITSFILE'], metadata[img]['JD'], chartime, flt, metadata[img]['EXPTIME'], metadata[img]['AIRMASS'], metadata[img]['IDNUM'], \
            zero[img], zerr[img], (metadata[img]['LIMFLX3']+zero[img]), ffit.wssrndf, tarid, obsid))



