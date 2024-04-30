#!/usr/bin/env python3

import os
import sys
import astropy
import astropy.wcs
import astropy.table
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
import argparse
import scipy.optimize as opt
#from kapteyn import kmpfit

def delin(arg):
    """cauchy delinearization to give outliers less weight and have more robust fitting"""
    try:
        ret=np.sqrt(np.log1p(arg**2))
    except RuntimeWarning:
#        print(str(arg))
        ret=np.sqrt(np.log1p(arg**2))
    return ret

# h(x,r)=r*sqrt(1+x*x/r/r)
# gnuplot> plot "20050717204005-238df.fits.xat" u 4:(log10($5)), h(x-B1,R)*dQ+(x-L1)/Q
# gnuplot> plot "20050717204005-238df.fits.xat" u 4:5, 10**(h(x-B1,R)*dQ+(x-L1)/Q)

def errormodel(params, data):
    """residuals used when fitting the magnitude limit"""
    x, y = data
    L1, Q = params
#    if fix25:
    return delin( y - (x-L1)/Q  )
#    else:
#        return delin(y-np.power(10.0,((x-L1)/Q)))

def delin(arg):
    """delinearization to give outliers less weight and have more robust fitting"""
    # linear (no delinearization)
    # return np.sqrt((r**2))

    # cauchy
    try:
        ret=np.sqrt(np.log1p(arg**2))
    except RuntimeWarning:
        print(str(arg))
        ret=np.sqrt(np.log1p(arg**2))
    return ret

    # arctan
    # return np.sqrt(np.arctan(r**2))

def load_chip_id(header):
    """standard chip ID to be always the same"""
    try:
        chip = header['CCD_SER']
    except:
        #return -1,"unknown"
        return "unknown"

    if chip == "":
        try:
            chip = header['CCD_TYPE']
        except:
            chip = "unknown"

    if chip == "":
        chip = "unknown"

    return chip


def readOptions(args=sys.argv[1:]):
  parser = argparse.ArgumentParser(description="Compute photometric calibration for a FITS image.")
  parser.add_argument("-v", "--verbose", action='store_true', help="Print debugging info.")
  parser.add_argument("-o", "--output", action='store_true', help="Output file.")
  parser.add_argument("-n", "--nonlin", help="CCD is not linear, apply linear correction on mag.", action='store_true')
  parser.add_argument("-f", "--filter", help="Override filter info from fits", type=str)
  parser.add_argument("files", help="Frames to process", nargs='+', action='extend', type=str)
  opts = parser.parse_args(args)
  return opts

options = readOptions(sys.argv[1:])


for arg in options.files:

    i_have_cat = False
    i_have_fits = False
    # either the argument may be a fits file or it may be an output of sextractor
    try:
        det = astropy.io.ascii.read(arg, format='sextractor')
        if options.verbose: print("Argument %s is a sextractor catalog"%(arg))
        i_have_cat = True
        arg_is_cat = True
    except:
        arg_is_cat = False
    
    # either the argument may be a fits file or it may be an output of sextractor
    try:
        fitsfile = astropy.io.fits.open(arg)
        if options.verbose: print("Argument %s is a fits file"%(arg))
        filef = arg
        i_have_fits = True
        arg_is_fits = True
    except:
        arg_is_fits = False

    if not (arg_is_cat or arg_is_fits): 
        if options.verbose: print("Argument %s is not a fits nor a sextractor catalog, pass"%(arg))
        continue
    
    if arg_is_fits:
        catf = arg + ".xat"
        try:
            det = astropy.io.ascii.read(catf, format='sextractor')
            i_have_cat = True
        except:
            catf = os.path.splitext(arg)[0] + ".cat"
            try:
                det = astropy.io.ascii.read(catf, format='sextractor')
                i_have_cat = True
            except:
                try:
                    catf = arg + ".xat"
                    print("Running sscat-noradec %s"%(arg))
                    os.system("sscat-noradec %s"%(arg))
                    det = astropy.io.ascii.read(catf, format='sextractor')
                    i_have_cat = True
                except:
                    print("%s: is an image but I found no sextractor output, pass"%(arg))
                    continue
        if options.verbose: print("Will use %s as a sextractor catalog"%(catf))
           
    if arg_is_cat:
        filef = os.path.splitext(arg)[0]
        try:
            fitsfile = astropy.io.fits.open(filef) 
            i_have_fits = True
        except:
            filef = filef + ".fits"
            try:
                fitsfile = astropy.io.fits.open(filef) 
                i_have_fits = True
            except:
                print("%s: is a sextractor output but I found no related fitsfile, pass"%(arg))
                continue
        if options.verbose: print("Will use %s as a fits file"%(filef))
    
    output = os.path.splitext(filef)[0] + ".det"
    if options.verbose: print("Will use %s as an output file"%(output))

#    if not (i_have_fits and i_have_cat):
#        print("%s: cannot open either catalog or a fitsfile"%(arg))
#        continue

    det.meta['FITSFILE'] = filef

    # remove zeros in the error column
    det['MAGERR_AUTO'] = np.sqrt(det['MAGERR_AUTO']*det['MAGERR_AUTO']+0.0005*0.0005)

    # === compute a detection limit from errorbars ===
    # get all the objects from the input file
    # fit it with:
    # error = 10**((x-L1)/2.5)
    # error = 10**((x-L1)/Q)
    # odhad L1 muze byt zeropoint zmenseny o ~5
    # fit [16:] 10**(-(L1-x)/2.5) "d50-n2.fits.xat" u ($4+22.881):5:5 via L1
    # a pak s-sigma limit bude:
    # l(s)=L1+Q*log10(1.091/s)

    #fix25=False
    fix25=True
    p0 = [1, 2.5]
    res = opt.least_squares(errormodel, p0, args=[(det['MAG_AUTO'],np.log10(det['MAGERR_AUTO']))] )
#    fitobj = kmpfit.Fitter(residuals=errormodel, data=(det['MAG_AUTO'],np.log10(det['MAGERR_AUTO'])), xtol=1e-13)
#    try:
#        fitobj.fit(params0=p0)
#    except RuntimeError:
#        print(arg,": Magnitude limit fit failed!")
#        continue
#    det.meta['LIMFLX3'] = fitobj.params[0]+fitobj.params[1]*np.log10(1.091/3)
#    det.meta['LIMFLX10'] = fitobj.params[0]+fitobj.params[1]*np.log10(1.091/30)
    det.meta['LIMFLX3'] = res.x[0]+res.x[1]*np.log10(1.091/3)
    det.meta['LIMFLX10'] = res.x[0]+res.x[1]*np.log10(1.091/30)

    # stability testing of these limits tested at a series of short exposures of AD Leo was 1-sigma=0.013 mag. 
    # level of accuracy of limits set this way depends on the undelying detection alorithm, its gain/rn settings and aperture
    if options.verbose:
        print(res)
    """
    if options.verbose:
        print("3-sigma limit = %.3f mag\n30-sigma limit = %.3f mag"%(det.meta['LIMFLX3'],det.meta['LIMFLX10']))
        print("\nFit limflux kmpfit output:")
        print("====================")
        print("Best-fit parameters:    ", fitobj.params)
        print("Asymptotic error:      ", fitobj.xerror)
        print("Error assuming red.chi^2=1: ", fitobj.stderr)
        print("Chi^2 min:         ", fitobj.chi2_min)
        print("Reduced Chi^2:       ", fitobj.rchi2_min)
        print("Iterations:         ", fitobj.niter)
        print("Number of free pars.:    ", fitobj.nfree)
        print("Degrees of freedom:     ", fitobj.dof, "\n")
    """

    c = fitsfile[0].header
    fitsfile.close()
    
    for i,j in c.items():
        if i == 'NAXIS' or i == 'NAXIS1' or i== 'NAXIS2' or i == 'BITPIX': continue
        if len(i)>8: continue
        if "." in i: continue
        det.meta[i] = j

    det.meta['CHIP_ID'] = load_chip_id(c)

    try:
        latitude = np.float64(c['LATITUDE'])
    except:
        try:
            latitude = np.float64(c['TEL_LAT'])
        except:
            latitude = 49.9090806 # BART Latitude
    det.meta['LATITUDE'] = latitude

    try:
        longitude = np.float64(c['LONGITUD'])
    except:
        try:
            longitude = np.float64(c['TEL_LONG'])
        except:
            longitude = 14.7819092 # BART Longitude
    det.meta['LONGITUD'] = longitude

    try:
        altitude = np.float64(c['ALTITUDE'])
    except:
        try:
            altitude = np.float64(c['TEL_ALT'])
        except:
            altitude = 530 # BART Altitude
    det.meta['ALTITUDE'] = altitude


    # get time of observations
    ctime = None
    try:
        time = Time(det.meta['DATE-OBS'])
        ctime = time.to_value('unix')
        if options.verbose:
            print("CTIME from DATE-OBS", ctime)
        det.meta['JD'] = 2440587.5 + ctime/86400.
    except:
        print("No DATE-OBS? That's BAD BAD BAD!")

    try:
        julian_date = np.float64(c['JD_START'])
    except KeyError:
        try:
            julian_date = np.float64(c['JD'])
        except:
            julian_date = 0

    det.meta['JD_START'] = julian_date

    try:
        julian_date_end = np.float64(c['JD_END'])
    except KeyError:
        try:
            julian_date_end = julian_date+np.float64(c['EXPTIME'])/86400
        except:
            try:
                julian_date_end = julian_date+np.float64(c['EXPOSURE'])/86400
            except:
                julian_date_end = 0

    det.meta['JD_END'] = julian_date_end

    try:
        usec = c['USEC']
    except:
        usec = 0

    try:
        ctime = np.float64(c['CT_START']) 
    except KeyError:
        try:
            ctime = np.float64(c['CTIME']) 
        except KeyError:
            ctime = 0

    tsec = np.float64(ctime+usec/1000000.)

    try:
        tsecend = np.float64(c['CT_END'])
    except KeyError:
        try:
            tsecend = tsec+np.float64(c['EXPTIME'])
        except KeyError:
            tsecend = 0

    tsecend = np.float64(ctime+usec/1000000.)

    # this is a wrong but far the most reliable solution (wrong:leap secs?)
    if tsec > 0:
        det.meta['JD'] = 2440587.5 + tsec/86400.
    if tsecend > 0:
        det.meta['JD_END'] = 2440587.5 + tsecend/86400.

    # if I do not set this, the following WCS load will complain
    det.meta['MJD-OBS'] = det.meta['JD'] - 2400000.5
#    print("JD="+str(det.meta['JD']),"JD_END="+str(det.meta['JD_END']));

   # det.meta['RADECSYSa']=det.meta['RADECSYS']
   # print(type(det.meta))
   # del det.meta['RADECSYS']

    if options.verbose:
        print("MJD-OBS=%.6f"%(det.meta['MJD-OBS']))

    imgwcs = astropy.wcs.WCS(det.meta)
    
    if options.filter != None:
        fits_fltr = options.filter
    else:
            try:
                fits_fltr = c['FILTER']
            except:
                fits_fltr = 'N'

            if fits_fltr == "OIII": fits_fltr = "oiii"
            if fits_fltr == "Halpha": fits_fltr = "halpha"
            if fits_fltr == "u": fits_fltr = "Sloan_u"
            if fits_fltr == "g": fits_fltr = "Sloan_g"
            if fits_fltr == "r": fits_fltr = "Sloan_r"
            if fits_fltr == "i": fits_fltr = "Sloan_i"
            if fits_fltr == "z": fits_fltr = "Sloan_z"
            if fits_fltr == "UNK": fits_fltr = "N"
            if fits_fltr == "C": fits_fltr = "N"
            if fits_fltr == "clear": fits_fltr = "N"

    det.meta['FILTER'] = fits_fltr

    # RTS2 names the object coordinates LOGICALLY!:
    try:
        det.meta['OBJRA'] = np.float64(c['ORIRA'])
    except:
        det.meta['OBJRA'] = -100

    try:
        det.meta['OBJDEC'] = np.float64(c['ORIDEC'])
    except:
        det.meta['OBJDEC'] = -100

    tmp = imgwcs.all_world2pix(det.meta['OBJRA'], det.meta['OBJDEC'], 0)

    if not np.isnan(tmp[0]):
        det.meta['OBJX'] = np.float64(tmp[0])
    if not np.isnan(tmp[1]):
        det.meta['OBJY'] = np.float64(tmp[1])
#    print("===",det.meta['OBJRA'],det.meta['OBJDEC'],"===")
#    print("===",tmp[0],tmp[1],"===")

    try:
        det.meta['AIRMASS'] = c['AIRMASS']
    except:
        det.meta['AIRMASS'] = -1

    try:
        exptime = c['EXPTIME']
    except:
        exptime = -1

    try:
        exptime__ = c['EXPOSURE']
        exptime = exptime__
    except:
        nic = 0

    det.meta['EXPTIME'] = exptime

    try:
        obsid_n = c['OBSID']
    except:
        obsid_n = 0

    try:
        obsid_d = c['SCRIPREP']
    except:
        obsid_d = 0

    det.meta['OBSID'] = "%0d.%02d"%(obsid_n, obsid_d)

    det.meta['CTRX'] = c['NAXIS1']/2
    det.meta['CTRY'] = c['NAXIS2']/2
    # NAXIS1/2 will not stay in the header
    det.meta['IMGAXIS1'] = c['NAXIS1']
    det.meta['IMGAXIS2'] = c['NAXIS2']
    rd = imgwcs.all_pix2world([[det.meta['CTRX'], det.meta['CTRY']], [0, 0], [det.meta['CTRX'], det.meta['CTRY']+1]], 0)
    #print(rd)
    
    det.meta['CTRRA'] = rd[0][0]
    det.meta['CTRDEC'] = rd[0][1]

    center = SkyCoord(rd[0][0]*astropy.units.deg, rd[0][1]*astropy.units.deg, frame='fk5')
    corner = SkyCoord(rd[1][0]*astropy.units.deg, rd[1][1]*astropy.units.deg, frame='fk5')
    pixel = SkyCoord(rd[2][0]*astropy.units.deg, \
        rd[2][1]*astropy.units.deg, frame='fk5').separation(center)
    field = center.separation(corner)
    if options.verbose:
        print("Field size: %.2f°"%(field.deg))
        print("Pixel size: %.2f\""%(pixel.arcsec))

    det.meta['FIELD'] = field.deg
    det.meta['PIXEL'] = pixel.arcsec

    try:
        fwhm = c['FWHM']
        if fwhm < 0.5: 
                fwhm = 2
    except:
        fwhm = 2

    if options.verbose:
        print("Detected FWHM %.1f pixels (%.1f\")"%(fwhm, fwhm*pixel.arcsec))

    det.meta['FWHM'] = fwhm

#    det.write(output, format="fits", overwrite=True)
    det.write(output, format="ascii.ecsv", overwrite=True)

