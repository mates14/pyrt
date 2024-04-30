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

def isnumber(a):
    try:
        k=int(a)
        return True
    except:
        return False

def exportColumnsForDS9(columns, file="ds9.reg", size=10, width=3, color="red"):
    some_file = open("noident.reg", "w+")
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
  parser.add_argument("-u", "--usno", help="Use USNO catalog.", action='store_true')
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
    return cat

def get_usno(rasc, decl, width=0.25, height=0.25, mlim=17):
    cat = None
    usno_ecsv_tmp = "usno%ld.ecsv"%(os.getpid())
    hx=2*height
    if hx>4: hx=4
    command = 'ubcone' + ' -P %f'%(rasc/15.0) + ' -p %f'%(decl) + ' -S %f -s %f'%(2*width, hx) + ' -i' + ' %.2f'%(mlim) + " -O" + usno_ecsv_tmp
    print(command)
    os.system(command)
    cat = astropy.io.ascii.read(usno_ecsv_tmp, format='ecsv')
    os.system("rm " + usno_ecsv_tmp)

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

imgno=0

def simple_color_model(line, data):

    mag,color1,color2,color3,color4=data
    #print(data)
    model=0
    for chunk in line.split(","):
        term,strvalue = chunk.split("=")
        value=np.float64(strvalue)
       # print(term,value)
        if term[0] == 'P':
            pterm = value; n=1;
            for a in term[1:]:
                if isnumber(a): n = int(a)
                if a == 'C': pterm *= np.power(color1, n); n=1;
                if a == 'D': pterm *= np.power(color2, n); n=1;
                if a == 'E': pterm *= np.power(color3, n); n=1;
                if a == 'F': pterm *= np.power(color4, n); n=1;
                if a == 'X' or a == 'Y' or a == 'R': pterm = 0;
            model += pterm 
        if term == 'XC': 
            if value < 0: bval = value * color1; 
            if value > 0 and value <= 1: bval = value * color2; 
            if value > 1: bval = (value-1) * color3 + color2; 
        #    print("***",value,bval,color1,color2,color3)
            model += bval;
    return mag+model

def open_det_file(arg, verbose=True):
    """Opens a file if possible, given .ecsv or .fits"""
    det = None
    
    fn = os.path.splitext(arg)[0] + ".ecsv"
    
    try:
        det = astropy.table.Table.read(arg)
        det.meta['filename'] = arg;
        return det
    except:
        det = None
        if verbose: print("%s did not open as a binary table"%(arg)); 

    if not os.path.isfile(fn):
        try:
            if verbose: print("%s does not seem to exist, will run cat2det.py to make it"%(fn));
            os.system("cat2det.py %s"%(arg))
        except:
            if verbose: print("Failed to run cat2det.py");

    try:
        det = astropy.table.Table.read(fn)
        det.meta['filename'] = fn;
        return det
    except:
        if verbose: print("%s did not open as a binary table"%(fn)); 
        det = None

    try:
        det = astropy.table.Table.read(fn, format="ascii.ecsv")
        det.meta['filename'] = fn;
        return det
    except:
        if verbose: print("%s did not open as an ecsv table"%(fn)); 
        det = None

    return det


for arg in options.files:

    det = open_det_file(arg, verbose=options.verbose) 
    if det is None: 
        if options.verbose: print("Cannot handle %s Skipping."%(arg))
        continue
    if options.verbose: print("Input file:", det.meta['filename'])

    remove_junk(det.meta)

    imgwcs = astropy.wcs.WCS(det.meta)
    det['ALPHA_J2000'], det['DELTA_J2000'] = imgwcs.all_pix2world( [det['X_IMAGE']], [det['Y_IMAGE']], 1)

    try:
        field=det.meta['FIELD']
    except:
        det.meta['FIELD'] = 180

    if options.enlarge is not None:
        enlarge = options.enlarge
    else: enlarge=1

    # 2000.0 = 2451544.5 
    # 2015.5 = 2457204.5 # reference epoch of Gaia DR2
    epoch = ( det.meta['JD'] - 2457204.5 ) / 365.2425 # Epoch for PM correction

    start = time.time()
    cat = get_atlas(det.meta['CTRRA'], det.meta['CTRDEC'], width=enlarge*det.meta['FIELD'], height=enlarge*det.meta['FIELD'], mlim=options.maglim)
    cat['radeg'] += epoch*cat['pmra']
    cat['decdeg'] += epoch*cat['pmdec']
    if options.usno:
        usno = get_usno(det.meta['CTRRA'], det.meta['CTRDEC'], width=enlarge*det.meta['FIELD'], height=enlarge*det.meta['FIELD'], mlim=options.maglim)
    if options.verbose: print("Catalog search took %.3fs"%(time.time()-start))

    if options.idlimit: idlimit = options.idlimit
    else: 
        try:
            idlimit = det.meta['FWHM']
            if options.verbose: print("idlimit set to fits header FWHM value of %f pixels."%(idlimit))
        except:
            idlimit = 2.0
            if options.verbose: print("idlimit set to a hard-coded default of %f pixels."%(idlimit))

    # ===  identification with KDTree  ===
    Y = np.array([det['X_IMAGE'], det['Y_IMAGE']]).transpose()

    start = time.time()
    try:
        Xt = np.array(imgwcs.all_world2pix(cat['radeg'], cat['decdeg'],1))
        X = Xt.transpose()
        if options.usno:
            Ut = np.array(imgwcs.all_world2pix(usno['radeg'], usno['decdeg'],1))
            U = Ut.transpose()
        # careful: Xselect conditions are inverted, i.e. what should not be left in
#        Xselect = np.any([np.isnan(Xt[0]),np.isnan(Xt[1]),Xt[0]<0,Xt[1]<0,Xt[0]>det.meta['IMAGEW'],Xt[1]>det.meta['IMAGEH'],cat['Sloan_r']>6.0], axis=0)
#        X = Xt.transpose()[~Xselect]
    except:
        if options.verbose: print("Astrometry of %s sucks! Skipping."%(arg))
        continue
    tree = KDTree(X)
    nearest_ind, nearest_dist = tree.query_radius(Y, r=idlimit, return_distance=True, count_only=False)
    if options.usno:
        tree_u = KDTree(U)
        nearest_ind_u, nearest_dist_u = tree_u.query_radius(Y, r=idlimit, return_distance=True, count_only=False)
    if options.verbose: print("Object cross-id took %.3fs"%(time.time()-start))

    some_file = open("noident.reg", "w+")
    some_file.write("# Region file format: DS9 version 4.1\nglobal color=yellow dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
    for cc in cat:
        some_file.write("circle(%.7f,%.7f,%.3f\")\n"%(cc["radeg"], cc["decdeg"],5))
    some_file.close()

    some_file = open("noidentXY.reg", "w+")
    some_file.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
    for xx, dd in zip(nearest_ind, det):
        if len(xx) < 1: 
            if dd["MAGERR_AUTO"] < 1.091/5:some_file.write("circle(%.7f,%.7f,%.3f\") # color=red\n"%(dd["X_IMAGE"], dd["Y_IMAGE"],1.5*idlimit*dd.meta['PIXEL']))
            else: some_file.write("circle(%.7f,%.7f,%.3f\") # color=black\n"%(dd["X_IMAGE"], dd["Y_IMAGE"],1.5*idlimit*dd.meta['PIXEL']))
        else: some_file.write("circle(%.7f,%.7f,%.3f\")\n"%(dd["X_IMAGE"], dd["Y_IMAGE"],1.5*idlimit*dd.meta['PIXEL']))
    some_file.close()

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

    some_file = open("noidentJK.reg", "w+")
    some_file.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
    # make pairs to be fitted
    for i, d in zip(nearest_ind, det):

        mdiff = 0
        match = None
        bestmatch = 1e99
        bestmag = 0
        magdet = np.float64(d['MAG_CALIB'])
        errdet = np.float64(d['MAGERR_CALIB'])

        for k in i:
            mag0 = cat[k]['Sloan_g']
            mag1 = cat[k]['Sloan_r']
            mag2 = cat[k]['Sloan_i']
            mag3 = cat[k]['Sloan_z']
            mag4 = cat[k]['J']

            magcat = mag1
            cm = simple_color_model( det.meta['RESPONSE'],
                (0,
                np.float64(mag0-mag1),
                np.float64(mag1-mag2),
                np.float64(mag2-mag3),
                np.float64(mag3-mag4)))
            
         #   if errdet<1.091/7: print(magdet,errdet,mag0,mag1,mag2,mag3,mag4,magcat) 

            mpar = (magcat-cm-magdet)/np.sqrt(errdet*errdet+0.01*0.01)
            if np.abs(mpar) < bestmatch:
                bestmatch = np.abs(mpar)
                bestmag = magcat-cm
                match = mpar
                mdiff = np.abs(magcat-cm-magdet)

        #print("o", match, bestmatch, bestmag, magdet, errdet)

        # pet moznosti: 
        # 1. objekt je v katalogu a neni detekovan (timto zpusobem to nedam)
       
        siglim = 5
        # 2. objekt neni v katalogu (= kandidat)
        if bestmag == 0 and errdet < 1.091/siglim:
            print("!",bestmag,magdet,errdet)
            some_file.write("circle(%.7f,%.7f,%.3f\") # color=red\n"%(d["X_IMAGE"], d["Y_IMAGE"],5*idlimit*d.meta['PIXEL']))
            continue

#        siglim = 3 # limit above which the star is considered deviant

        # 3. objekt je jasnejsi nez v katalogu (= kandidat)
        if match is not None and match > siglim and errdet<1.091/siglim and mdiff>0.05:
            print("-",bestmag,cm,magdet,errdet,np.abs(bestmag-magdet)/errdet,mdiff)
            some_file.write("circle(%.7f,%.7f,%.3f\") # color=yellow\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))

        # 4. objekt je slabsi nez v katalogu (= zajimavost)
        if match is not None and match < -siglim and errdet<1.091/siglim and mdiff>0.05:
            print("+",bestmag,cm,magdet,errdet,np.abs(bestmag-magdet)/errdet,mdiff)
            some_file.write("circle(%.7f,%.7f,%.3f\") # color=blue\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))
        
        # 5. objekt odpovida katalogu (nic)
        if match is not None and (( match > -siglim and (bestmag-magdet)/errdet < siglim) or mdiff<0.05 ) and errdet<1.091/siglim:
            some_file.write("circle(%.7f,%.7f,%.3f\") # color=green\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))
    #        print("o",bestmag,cm,magdet,errdet,np.abs(bestmag-magdet)/errdet)

    if options.usno:

        for i, dd in zip(nearest_ind_u, det):
            bestmatch = 1e99
            bestmag = None
            magdet = np.float64(dd['MAG_CALIB'])
            errdet = np.float64(dd['MAGERR_CALIB'])

            for k in i:
                mag_usno = 0
                nmag_usno = 0 
                for uflt in ['R1','R2']:
                    if usno[k][uflt] != 0: 
                        mag_usno += usno[k][uflt]
                        nmag_usno += 1
                if nmag_usno > 1: 
                    magcat = mag_usno / nmag_usno

                if magcat != 0:
                    mpar = (magcat-magdet)/np.sqrt(errdet*errdet+0.3*0.3)
                    if np.abs(mpar) < bestmatch:
                        bestmatch = np.abs(mpar)
                        bestmag = magcat

                    if mpar < 5 and mpar > -5:
                        match = True

            usiglim = siglim

            if bestmag is None and errdet < 1.091/usiglim:
                print("!!u",bestmag,magdet,errdet)
                some_file.write("circle(%.7f,%.7f,%.3f\") # color=magenta width=3\n"%(dd["X_IMAGE"], dd["Y_IMAGE"], 5.5*idlimit*d.meta['PIXEL']))
            else:
                if errdet < 1.091/ usiglim:
                    some_file.write("circle(%.7f,%.7f,%.3f\") # color=cyan width=3\n"%(dd["X_IMAGE"], dd["Y_IMAGE"], 1.25*idlimit*d.meta['PIXEL']))
                    
    some_file.close()


    # === fields are set up ===
