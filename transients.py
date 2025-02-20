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
from sklearn.neighbors import KDTree,BallTree

import zpnfit
import fotfit

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
  parser.add_argument("-f", "--frame", help="Image frame width to be ignored in pixels (default=10)", type=float)
  parser.add_argument("-g", "--regs", action='store_true', help="Save per image regs")
  parser.add_argument("-s", "--siglim", help="Sigma limit for detections to be taken into account.", type=float)
  parser.add_argument("-m", "--min-found", help="Minimum number of occurences to consider candidate valid", type=int, default=4)
  parser.add_argument("-u", "--usno", help="Use USNO catalog.", action='store_true')
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

    mag,color1,color2,color3,color4=data
    #print(data)
    model=0
    try:
        for chunk in line.split(","):
            term,strvalue = chunk.split("=")
            if term == 'FILTER': continue
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
    except ValueError:
        model=0
    return mag+model

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

if options.frame is not None: frame=options.frame
else: frame=10

if options.siglim is not None: siglim = options.siglim
else: siglim = 5


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
        if det.meta['CTIME']+det.meta['EXPTIME']/2-t0 > 43200:
            continue

    if det.meta['CTIME'] < mintime: mintime = det.meta['CTIME']
    if det.meta['CTIME']+det.meta['EXPTIME'] > maxtime: maxtime = det.meta['CTIME']+det.meta['EXPTIME']
    imgtimes.append(det.meta['CTIME']+det.meta['EXPTIME']/2)

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

    if options.idlimit: idlimit = options.idlimit
    else:
        try:
            idlimit = det.meta['FWHM']
            if options.verbose: print("idlimit set to fits header FWHM value of %f pixels."%(idlimit))
        except:
            idlimit = 1.2
            if options.verbose: print("idlimit set to a hard-coded default of %f pixels."%(idlimit))

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
    tree = KDTree(X)
    nearest_ind, nearest_dist = tree.query_radius(Y, r=idlimit, return_distance=True, count_only=False)
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

            mpar = (magcat-cm-magdet)/np.sqrt(errdet*errdet+0.01*0.01)
            if np.abs(mpar) < bestmatch:
                bestmatch = np.abs(mpar)
                bestmag = magcat-cm
                match = mpar
                mdiff = np.abs(magcat-cm-magdet)

        # pet moznosti:
        # 1. objekt je v katalogu a neni detekovan (timto zpusobem to nedam)

        # 2. objekt neni v katalogu (= kandidat)
        if bestmag == 0 and errdet < 1.091/siglim:
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

            if d['X_IMAGE'] < frame or d['Y_IMAGE']<frame or d['X_IMAGE'] > d.meta['IMGAXIS1']-frame or d['Y_IMAGE']>d.meta['IMGAXIS2']-frame:
                continue

            candx.append(d["X_IMAGE"])
            candy.append(d["Y_IMAGE"])
            candra.append(d['ALPHA_J2000'])
            canddec.append(d['DELTA_J2000'])
            candtime.append(d.meta['CTIME'])
            candexp.append(d.meta['EXPTIME'])
            candmag.append(d['MAG_CALIB'])
            canddmag.append(d['MAGERR_CALIB'])
            candfw.append(d['FWHM_IMAGE'])
            continue

        # 3. objekt je jasnejsi nez v katalogu (= kandidat)
        if match is not None and match > siglim and errdet<1.091/siglim and mdiff>0.05:
#            print("+",bestmag,cm,magdet,errdet,np.abs(bestmag-magdet)/errdet,mdiff)
            if options.regs:
                some_file.write("circle(%.7f,%.7f,%.3f\") # color=yellow\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))

        # 4. objekt je slabsi nez v katalogu (= zajimavost)
        if match is not None and match < -siglim and errdet<1.091/siglim and mdiff>0.05:
#            print("-",bestmag,cm,magdet,errdet,np.abs(bestmag-magdet)/errdet,mdiff)
            if options.regs:
                some_file.write("circle(%.7f,%.7f,%.3f\") # color=blue\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))

        # 5. objekt odpovida katalogu (nic)
        if match is not None and (( match > -siglim and (bestmag-magdet)/errdet < siglim) or mdiff<0.05 ) and errdet<1.091/siglim:
            if options.regs:
                some_file.write("circle(%.7f,%.7f,%.3f\") # color=green\n"%(d["X_IMAGE"], d["Y_IMAGE"],1.5*idlimit*d.meta['PIXEL']))
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
        usno_phases = [
            ('simple', options.maglim, idlimit, 0),
            ('double', options.maglim - 1, 4, 1),
            ('bright', options.maglim - 9, 10, 0)
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
    cand = astropy.table.Table([candra,canddec,candtime,candexp,candmag,canddmag,candfw,np.int64(np.ones(len(candra)))], \
        names=['ALPHA_J2000','DELTA_J2000','CTIME','EXPTIME','MAG_CALIB','MAGERR_CALIB','FWHM_IMAGE','NUM'])
#    print("file", os.path.splitext(arg)[0], len(cand), "candidates")

    if len(candra)>1:
    # remove doubles!
#            tree = KDTree( np.array([cand['ALPHA_J2000'], cand['DELTA_J2000']]).transpose())
            tree = BallTree( np.array([cand['ALPHA_J2000']*np.pi/180, cand['DELTA_J2000']*np.pi/180]).transpose(), metric='haversine')
            nearest_ind, nearest_dist = tree.query_radius( np.array([cand['ALPHA_J2000']*np.pi/180, cand['DELTA_J2000']*np.pi/180]).transpose() , r=d.meta['PIXEL']*idlimit/3600.*2*np.pi/180, return_distance=True, count_only=False)
            i=0;
            doubles=[]
            for g,h in zip(nearest_ind,nearest_dist):
                for j,k in zip(g,h):
                    if j < i:
                        doubles.append([j])
                        print(f"candidate {i} is a double to {j} (distance={3600*180*k/np.pi}\") and will be removed");
                i+=1;
            cand.remove_rows(doubles)

#    if imgno == 0 or len(old)<1:
    if len(old)<1:
        old = cand
        for i in cand:
            mags.append(astropy.table.Table(i))

    if imgno > 0 and len(cand)>0:
        tree = BallTree( np.array([old['ALPHA_J2000']*np.pi/180, old['DELTA_J2000']*np.pi/180]).transpose(), metric='haversine')
        nearest_ind, nearest_dist = tree.query_radius( np.array([cand['ALPHA_J2000']*np.pi/180, cand['DELTA_J2000']*np.pi/180]).transpose() , \
            r=d.meta['PIXEL']*idlimit/3600.0*np.pi/180*2,\
            return_distance=True, count_only=False)

        for bu,ba,bo in zip(cand, nearest_ind, nearest_dist):
            if len(ba)>0:
                # wrap-around non-safe! (and does not work!)
#                ora=old['ALPHA_J2000'][ba[0]]
#                odec=old['DELTA_J2000'][ba[0]]
#                old['ALPHA_J2000'] = (old['ALPHA_J2000'][ba[0]] * old['NUM'][ba[0]] + bu['ALPHA_J2000'])/(old['NUM']+1)
#                old['DELTA_J2000'] = (old['DELTA_J2000'][ba[0]] * old['NUM'][ba[0]] + bu['DELTA_J2000'])/(old['NUM']+1)
                old['NUM'][ba[0]] += 1
                mags[ba[0]].add_row(bu)
            else: # this is a new position
                old.add_row(bu)
                mags.append(astropy.table.Table(bu))

    imgno+=1

if len(old) < 1:
        print("No transients found (old is None)")
        sys.exit(0)

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

transtmp = []
for oo,i in zip(mags,range(0,len(mags))):
        # full rejection if less than min_found
        num_found = len(oo)
        if num_found < options.min_found: continue

        status = list("!!!!") # start with good (movement in pixels, movement significant, magvar > 3)
        t0=np.min(old['CTIME'])/2+np.max(old['CTIME'])/2
        dt=np.max(old['CTIME'])-np.min(old['CTIME'])
        x = oo['CTIME']-t0
        y = oo["ALPHA_J2000"]
        z = oo["DELTA_J2000"]

        fwhm_mean=np.average(oo["FWHM_IMAGE"])

        data = [x,y]
        fitvalues = [oo['ALPHA_J2000'][0],1.0] # [a0,da]
        res = fit.least_squares(movement_residuals, fitvalues, args=[data], ftol=1e-14)
        a0=res.x[0]
        da=res.x[1]*3600*3600 # -> arcsec/h
        sa = np.median(np.abs(movement_residuals(res.x, data))) / 0.67
        data = [x,z]
        fitvalues = [oo['DELTA_J2000'][0],1.0] # [a0,da]
        res = fit.least_squares(movement_residuals, fitvalues, args=[data], ftol=1e-14)
        d0=res.x[0]
        dd=res.x[1]*3600*3600 # -> arcsec/h
        sd = np.median(np.abs(movement_residuals(res.x, data))) / 0.67
        dpos = np.sqrt(dd*dd+da*da*np.cos(d0*np.pi/180.0)*np.cos(d0*np.pi/180.0))* (dt/3600.0)
        sigma = np.sqrt(sd*sd+sa*sa*np.cos(d0*np.pi/180.0)*np.cos(d0*np.pi/180.0))
        if dpos > d.meta['PIXEL']: # PIXEL is in arcsec, so "is the movement more than a pixel during the sequence?"
            status[0] = "p"
        if dpos > 2*d.meta['PIXEL']: # PIXEL is in arcsec, so "is the movement more than a pixel during the sequence?"
            status[0] = "P"
        if dpos > 2*sigma: # is the movement significant?
            status[1] = "m"
        if dpos > 3*sigma: # is the movement significant?
            status[1] = "M"
        cov = np.linalg.inv(res.jac.T.dot(res.jac))
        fiterrors = np.sqrt(np.diagonal(cov))

        mag0 = np.sum(oo['MAG_CALIB']*oo['MAGERR_CALIB'])/np.sum(oo['MAGERR_CALIB'])
        magvar = np.sqrt(np.average(np.power( (oo['MAG_CALIB']-mag0)/oo['MAGERR_CALIB'] ,2)))
        if magvar < 3: status[2]="v"
        if magvar < 2: status[2]="V"

# det.meta['FWHM'] musi byt nahrazeny prumerem snimku nebo lepe to udelat uplne jinak
#        if fwhm_mean < det.meta['FWHM']/2 or fwhm_mean > det.meta['FWHM']*2:
#            status[3]="F"
#        if fwhm_mean < det.meta['FWHM']/1.5 or fwhm_mean > det.meta['FWHM']*1.5:
#            status[3]="f"

        print("".join(status), f"num:{num_found} mag:{mag0:.2f} magvar:{magvar:.1f}, mean_pos:{a0:.5f},{d0:.5f}, movement: {da*np.cos(d0*np.pi/180.0):.3f},{dd:.3f}, sigma: {sa*3600:.2f},{sd*3600:.2f} fwhm_mean: {fwhm_mean}")

        newtrans = [np.int64(i),np.int64(num_found),a0,d0,da,dd,dpos,sa*3600,sd*3600,sigma*3600,mag0,magvar]
        transtmp.append(newtrans)

trans=astropy.table.Table(np.array(transtmp),\
        names=['INDEX','NUM','ALPHA_J2000','DELTA_J2000','ALPHA_MOV','DELTA_MOV','DPOS','ALPHA_SIG','DELTA_SIG','SIGMA','MAG_CALIB','MAG_VAR'],\
        dtype=['int64','int64','float64','float64','float32','float32','float32','float32','float32','float32','float32','float32'])
print(trans)

print("In total",len(old),"positions considered.")

# AND NOW SOME REPORTING
# MAKE A DS9 .REG FILE
regfile = "transients.reg"
some_file = open(regfile, "w+")
some_file.write("# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=3 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n")
for oo in trans:
#    print(oo)
    print("any  ra/dec %.7f %.7f num: %.0f"%(np.average(oo["ALPHA_J2000"]), np.average(oo["DELTA_J2000"]),len(mags[oo['INDEX']])))
    some_file.write("circle(%.7f,%.7f,%.3f\") # color=blue\n"%(np.average(oo["ALPHA_J2000"]), np.average(oo["DELTA_J2000"]),5*idlimit*d.meta['PIXEL']))
    if oo['SIGMA']<1.5 and oo['DPOS']/d.meta['PIXEL']<1.5 and oo['MAG_VAR'] > 2:
        print("stationary  ra/dec %.7f %.7f num: %.0f"%(np.average(oo["ALPHA_J2000"]), np.average(oo["DELTA_J2000"]),len(mags[oo['INDEX']])))
        some_file.write("circle(%.7f,%.7f,%.3f\") # color=red\n"%(np.average(oo["ALPHA_J2000"]), np.average(oo["DELTA_J2000"]),5*idlimit*d.meta['PIXEL']))
    if oo['SIGMA']<1.5 and oo['DPOS']/d.meta['PIXEL']>=1.5 and len(mags[oo['INDEX']]) > 7:
        print("moving      ra/dec %.7f %.7f num: %.0f"%(np.average(oo["ALPHA_J2000"]), np.average(oo["DELTA_J2000"]),len(mags[oo['INDEX']])))
        some_file.write("circle(%.7f,%.7f,%.3f\") # color=cyan\n"%(np.average(oo["ALPHA_J2000"]), np.average(oo["DELTA_J2000"]),5*idlimit*d.meta['PIXEL']))
some_file.close()

# GNUPLOT LIGHTCURVE
t0 = try_grbt0(d.meta['TARGET'])
some_file = open("transients.gp", "w+")
some_file.write("set yrange reverse\n")
some_file.write(f"set title \"{try_tarname(d.meta['TARGET'])}\"\n")
some_file.write("set terminal png\n")
some_file.write("set logs x\n")

#some_file.write("set mxtics (")
#some_file.write(")\n")

some_file.write("set xtics (")
for j in range(0,6):
        tic = np.power(10, np.log10(mintime-t0) + j/5.0*(np.log10(maxtime-t0) - np.log10(mintime-t0))  )
        some_file.write(f"\"{tic:.0f}\" {tic:.3f},")
for j in imgtimes:
        some_file.write(f" \"\" {j-t0},")
some_file.write(")\n")

some_file.write(f"set output \"transients-{d.meta['TARGET']}.png\"\n")
some_file.write(f"plot [{mintime-t0:.3f}:{maxtime-t0:.3f}] \\\n")

j=0;
for oo in trans:
    if oo['SIGMA']<1.5 and oo['DPOS']/d.meta['PIXEL']<1.5 and oo['MAG_VAR'] > 3:
        some_file.write(f"\"-\" u ($1+$3/2.):2:($3/2.0):4 w xye pt 7 t \"a:{oo['ALPHA_J2000']:.5f} d:{oo['DELTA_J2000']:.5f} v={oo['MAG_VAR']:.1f} s={oo['SIGMA']:.2f} p={oo['DPOS']/d.meta['PIXEL']:.2f}\",\\\n")
        j+=1
some_file.write("\n")

for oo in trans:
    if oo['SIGMA']<1.5 and oo['DPOS']/d.meta['PIXEL']<1.5 and oo['MAG_VAR'] > 3:
        for mag in mags[oo['INDEX']]:
            some_file.write("%ld %.3f %d %.3f\n"%(\
                mag['CTIME']-t0,\
                mag['MAG_CALIB'],\
                mag['EXPTIME'],\
                mag['MAGERR_CALIB']))
        some_file.write("e\n")

some_file.close()
if j>0:
    os.system("gnuplot < transients.gp")
#os.system("rm transients.gp")
# FINISHED GNUPLOT

#print(mags)
# === fields are set up ===
