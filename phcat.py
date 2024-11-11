#!/usr/bin/python3

import os
import sys
import argparse
import numpy as np

import astropy.wcs
import astropy.table
import astropy.io.fits

def read_options(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Compute photometric calibration for a FITS image.")
    parser.add_argument("-a", "--aperture", help="Override an automated aperture choice", type=float)
    parser.add_argument("-I", "--noiraf", help="Do not use IRAF", action='store_true')
    parser.add_argument("-b", "--background", help="Save the background check image", action='store_true')
    parser.add_argument("files", help="Frames to process", nargs='+', action='extend', type=str)
    opts = parser.parse_args(args)
    return opts

def gauss(r, sigma):
    '''gauss exponential'''
    return 1./sigma/np.sqrt(2*np.pi) * np.exp( -np.power(r,2)/2/sigma/sigma )

def do_matrix(file, fwhm):
    '''Generate sextractor convolution matrix of a given FWHM'''
    sigma = fwhm / np.sqrt(8.0*np.log(2.0))
    q = int(sigma*3+0.5)
#    base = os.path.splitext(file)[0]
    some_file = open(file, "w+")
    some_file.write("CONV NORM\n")
    some_file.write(f"# {2*q+1}x{2*q+1} convolution mask with FWHM = {fwhm} pixels.\n")
    for x in range(-q,q+1):
        for y in range(-q,q+1):
            wx = 100*gauss(np.sqrt(np.float32(x)*np.float32(x)+np.float32(y)*np.float32(y)),sigma)
            some_file.write(f"{wx:.1f} ")
        some_file.write("\n")
    some_file.close()
#    os.system(f"cat {file}")


def call_sextractor(file, fwhm, bg=False):
    """send sextractor to a file
    for a completely unknown file, this may better be a two pass process:
    first to find the FWHM, second to have proper sky and convolution settings for that FWHM"""
    base = os.path.splitext(file)[0]
    some_file = open(base+".sex", "w+")
    some_file.write( "CATALOG_TYPE     ASCII_HEAD\n")
#    some_file.write( "VERBOSE_TYPE     QUIET\n")
    some_file.write( "DETECT_THRESH 1\n")
    some_file.write( "ANALYSIS_THRESH 1\n")
    some_file.write(f"BACK_SIZE  {int(fwhm*1.5)+1}\n")
    some_file.write(f"BACK_FILTERSIZE  3\n")
    #some_file.write(f"BACK_SIZE 5\n")
    #some_file.write("BACK_FILTERSIZE  1\n")
    some_file.write(f"PARAMETERS_NAME  {base}.param\n")
    some_file.write(f"FILTER_NAME      {base}.conv\n")
    some_file.write(f"CATALOG_NAME     {base}.cat\n")
    # place this under some cmd-line option...
    if bg:
        some_file.write(f"checkimage_name  {base}-bg.fits\n")
        some_file.write( "checkimage_type  background\n")
    some_file.close()
    os.system(f'cat {base}.sex')

    base = os.path.splitext(file)[0]
    some_file = open(base+".param", "w+")
    some_file.write("NUMBER\n")
    some_file.write("ALPHA_J2000\n")
    some_file.write("DELTA_J2000\n")
    some_file.write("MAG_AUTO\n")
    some_file.write("MAGERR_AUTO\n")
    some_file.write("X_IMAGE\n")
    some_file.write("Y_IMAGE\n")
    some_file.write("ERRX2_IMAGE\n")
    some_file.write("ERRY2_IMAGE\n")
    some_file.write("FWHM_IMAGE\n")
    some_file.write("ELLIPTICITY\n")
    some_file.write("FLAGS\n")
    some_file.close()

    do_matrix(base+".conv", fwhm)

    os.system(f"sex -c {base}.sex {file}")
    det = astropy.io.ascii.read(base+".cat", format='sextractor')
    os.system(f'rm {base}.cat {base}.conv {base}.sex {base}.param')

    det.meta['IMAGEW']=astropy.io.fits.getval(file, "NAXIS1", 0)
    det.meta['IMAGEH']=astropy.io.fits.getval(file, "NAXIS2", 0)

#    os.system(f'rm {base}.conv {base}.sex {base}.param')
    return det

def try_target(file):
    '''Try to get RTS2 target coordinates for the given file'''
    try:
        hdr = astropy.io.fits.getheader(file)
        x, y = astropy.wcs.WCS(hdr).all_world2pix( [hdr['ORIRA']], [hdr['ORIDEC']], 1)
        return x[0],y[0]
    except:
        return None,None

def call_iraf(file, det):
    """call iraf/digiphot/daophot/phot on a file"""
    base = os.path.splitext(file)[0]

    fwhm = get_fwhm_from_detections(det)
    #fwhm = np.median(det[ det['MAGERR_AUTO'] < 1.091/10 ]['FWHM_IMAGE'])
    if np.isnan(fwhm):
        fwhm = np.nanmedian(det)
    print(f'FWHM={fwhm}')

    some_file = open(base+".coo.1", "w+")

    orix, oriy = try_target(file)
    print(orix,oriy)
    if orix is not None and oriy is not None and not np.isnan(orix) and not np.isnan(oriy):
        some_file.write(f"{orix:.3f} {oriy:.3f}\n")
        target_not_valid = False
    else:
        target_not_valid = True

    for x,y in zip(det['X_IMAGE'],det['Y_IMAGE']):
        if target_not_valid or np.sqrt((x-orix)*(x-orix)+(y-oriy)*(y-oriy)) > fwhm:
            some_file.write(f"{x:.3f} {y:.3f}\n")
    some_file.close()

    # now the FWHM is a good idea for large stars, but is should be enlarged
    # once the stars are too sharp, the sub-sqrt will make the transition
    # smooth so when comparing various images, there is no sharp edge between
    # groups
    ape=np.sqrt(fwhm*fwhm+4)
    danu=1.5*ape
    anu=2.0*ape

    # D50 Andor gain and rnoise, this stuff needs to be seriously improved
    try: ncombine = astropy.io.fits.getval(file, "NCOMBINE")
    except: ncombine = 1.0
    epadu=0.81 * ncombine
    rnoise=4.63 / np.sqrt(ncombine)

    some_file = open(base+".cl", "w+")
    some_file.write("noao\n")
    some_file.write("digiphot\n")
    some_file.write("daophot\n")
    some_file.write(f"phot {file} {base}.coo.1 {base}.mag.1")
    some_file.write(f" readnoi={rnoise} epadu={epadu}")
    some_file.write(f" calgori=centroid cbox={fwhm/2}")
    #some_file.write(f" calgori=none")
    some_file.write(f" salgori=mode annulus={anu} dannulu={danu}")
    some_file.write(f" apertur={ape} zmag=0 sigma=0 veri- datamax=60000")
    some_file.write("\n\n\n\n")
    some_file.write("logout\n")
    some_file.close()
    os.system(f"irafcl < {base}.cl > /dev/null")
    os.system(f'rm {base}.cl')
    os.system(f'rm {base}.coo.1')


    mag = astropy.io.ascii.read(base+".mag.1", format='daophot')
    #mag['MAG'] = mag['MAG'] - mag.meta['ZMAG']
    os.system(f'rm {base}.mag.1')
    return mag

def get_fwhm_from_detections(det, min_good_detections=30):
    """
    Calculate FWHM from detections using a two-tier approach:
    1. Try detections with good magnitude errors
    2. If insufficient, fall back to brightest objects
    
    Parameters:
    det: astropy.table.Table - Detection table from sextractor
    min_good_detections: int - Minimum number of detections needed before falling back
    
    Returns:
    float - Median FWHM value
    """

    sel = np.all( [ det['X_IMAGE'] < det.meta['IMAGEW']-32, det['Y_IMAGE'] < det.meta['IMAGEH']-32, det['X_IMAGE'] > 32, det['Y_IMAGE'] > 32 ], axis=0) 
    #print(sel, det, len(sel), len(det))
    det2 = det [ sel ]
    # First try: all detections with good magnitude errors
    good_detections = det2[det2['MAGERR_AUTO'] < 1.091/10]
    
    if len(good_detections) >= min_good_detections:
        return np.median(good_detections['FWHM_IMAGE'])
    
    # Second try: use 30 brightest objects
    # Sort by magnitude (lower is brighter)
    bright_detections = det2[np.argsort(det['MAG_AUTO'])[:30]]

    return np.median(bright_detections['FWHM_IMAGE'])


def main():
    '''Take over the world'''
    options = read_options(sys.argv[1:])
    for file in options.files:
        base = os.path.splitext(file)[0]
        det = call_sextractor(file, 2.0)
        new_fwhm = get_fwhm_from_detections(det)
        if not np.isnan(new_fwhm):
            det = call_sextractor(file, new_fwhm, bg=options.background)

        if options.noiraf:
            tbl = det[np.all([det['FLAGS'] == 0, det['MAGERR_AUTO']<1.091/2],axis=0)]
        else:
            mag = call_iraf(file, det)
            # merge det+mag
            det.rename_columns(('MAG_AUTO','MAGERR_AUTO'), ('MAG_SEX','MAGERR_SEX'))
            det.remove_columns(('X_IMAGE','Y_IMAGE','ERRX2_IMAGE','ERRY2_IMAGE'))
            mag.rename_columns(('ID','XCENTER','YCENTER','XERR','YERR','MAG','MERR'),\
                    ('NUMBER','X_IMAGE','Y_IMAGE','ERRX2_IMAGE','ERRY2_IMAGE','MAG_AUTO','MAGERR_AUTO'))
            mag.remove_columns(('XAIRMASS','IFILTER','XINIT','YINIT','IMAGE','COORDS','LID',\
                    'XSHIFT','YSHIFT','CERROR','SERROR','ITIME','OTIME','PERROR',\
                    'CIER','MSKY','STDEV','SSKEW','NSKY','NSREJ','SIER','SUM'
                    ))
            tbl = astropy.table.join(det, mag, keys='NUMBER')
            # table.write...
            tbl = tbl[np.all([tbl['PIER'] == 0,tbl['FLUX'] > 0,tbl['MAGERR_AUTO']<1.091/2],axis=0)]
        tbl.write(base+".cat", format="ascii.ecsv", overwrite=True)
        print(f'OBJECTS={len(tbl)}')

# this way, the variables local to main() are not globally available, avoiding some programming errors
if __name__ == "__main__":
    main()
